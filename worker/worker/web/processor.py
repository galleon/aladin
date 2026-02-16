"""
Web processor: orchestrates crawler, chunking, embedding, Qdrant storage.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from shared.config import settings, truncate_for_embedding
from shared.schemas import IngestionJobStatus, DocumentChunk
from shared.telemetry import get_tracer, IngestionMetrics

from .crawler import crawl_pages

logger = logging.getLogger(__name__)


class WebProcessor:
    """Processes web content for ingestion."""

    def __init__(self, job_ctx, job_id: str, collection_name: str):
        self.job_ctx = job_ctx
        self.job_id = job_id
        self.collection_name = collection_name
        self.tracer = get_tracer()
        self.metrics = IngestionMetrics()
        self.embedding_model_id = settings.EMBEDDING_MODEL

        self.embedding_client = OpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_API_BASE,
        )
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process(
        self,
        url: str,
        source_config: dict,
        processing_config: dict,
        metadata: dict,
    ) -> dict:
        with self.tracer.start_as_current_span("web_process") as span:
            span.set_attribute("url", url)
            start_time = time.time()

            await self._ensure_collection()

            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.CRAWLING,
                progress=10,
                message="Starting web crawl...",
            )

            crawl_start = time.time()
            pages = await crawl_pages(
                url,
                source_config,
                processing_config,
                metrics_callback=lambda n: self.metrics.pages_crawled.add(n, {"collection": self.collection_name}),
            )
            crawl_duration = (time.time() - crawl_start) * 1000

            self.metrics.record_processing_time(crawl_duration, "crawl")
            span.set_attribute("pages_crawled", len(pages))

            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.PARTITIONING,
                progress=40,
                message=f"Processing {len(pages)} pages...",
                pages_total=len(pages),
            )

            partition_start = time.time()
            all_chunks = []
            for i, page in enumerate(pages):
                chunks = await self._process_page(page, metadata)
                all_chunks.extend(chunks)
                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.PARTITIONING,
                    progress=40 + int(30 * (i + 1) / len(pages)) if pages else 40,
                    pages_processed=i + 1,
                )

            partition_duration = (time.time() - partition_start) * 1000
            self.metrics.record_processing_time(partition_duration, "partition")

            if not all_chunks:
                logger.warning("No content extracted from %s", url)
                return {"chunks_created": 0, "pages_crawled": len(pages)}

            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EMBEDDING,
                progress=75,
                message=f"Embedding {len(all_chunks)} chunks...",
            )

            embed_start = time.time()
            await self._embed_and_store(all_chunks)
            embed_duration = (time.time() - embed_start) * 1000

            self.metrics.record_processing_time(embed_duration, "embedding")
            self.metrics.record_chunks_created(len(all_chunks), self.collection_name)

            total_duration = (time.time() - start_time) * 1000
            span.set_attribute("chunks_created", len(all_chunks))
            span.set_attribute("total_duration_ms", total_duration)

            return {
                "chunks_created": len(all_chunks),
                "pages_crawled": len(pages),
                "crawl_duration_ms": crawl_duration,
                "partition_duration_ms": partition_duration,
                "embedding_duration_ms": embed_duration,
                "total_duration_ms": total_duration,
            }

    async def _ensure_collection(self):
        collections = self.qdrant.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        if not exists:
            try:
                test_embedding = self.embedding_client.embeddings.create(
                    input=["test"], model=self.embedding_model_id,
                )
                vector_size = (
                    len(test_embedding.data[0].embedding)
                    if test_embedding.data
                    else settings.EMBEDDING_DIMENSION
                )
            except Exception as e:
                vector_size = settings.EMBEDDING_DIMENSION
                logger.warning("Failed to detect embedding dimension: %s", e)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    async def _process_page(self, page: dict, metadata: dict) -> List[DocumentChunk]:
        chunks = []
        content = page.get("markdown", "")
        if not content:
            return chunks

        for table in page.get("tables", []):
            if table.get("markdown"):
                content += f"\n\n{table['markdown']}"

        texts = self.text_splitter.split_text(content)
        for i, text in enumerate(texts):
            chunks.append(
                DocumentChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=text,
                    metadata={
                        **metadata,
                        "source_type": "web",
                        "title": page.get("title", ""),
                        "crawl_depth": page.get("depth", 0),
                    },
                    source_url=page.get("url"),
                    chunk_index=i,
                    total_chunks=len(texts),
                )
            )
        return chunks

    async def _embed_and_store(self, chunks: List[DocumentChunk]):
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [truncate_for_embedding(c.content) for c in batch]
            response = self.embedding_client.embeddings.create(
                model=settings.EMBEDDING_MODEL, input=texts,
            )
            embeddings = [e.embedding for e in response.data]
            points = [
                PointStruct(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id),
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "source_url": chunk.source_url,
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "chunk_id": chunk.id,
                        **chunk.metadata,
                    },
                )
                for chunk, embedding in zip(batch, embeddings)
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            logger.info("Stored %d chunks in %s", len(points), self.collection_name)

    async def crawl_and_chunk(
        self,
        url: str,
        source_config: dict,
        processing_config: dict,
        metadata: dict | None = None,
    ) -> List[DocumentChunk]:
        """Crawl and chunk only (no embedding/storage). For CLI use."""
        metadata = metadata or {}
        pages = await crawl_pages(url, source_config, processing_config)
        all_chunks = []
        for page in pages:
            chunks = await self._process_page(page, metadata)
            all_chunks.extend(chunks)
        return all_chunks
