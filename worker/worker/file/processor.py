"""
File processor: orchestrates loader, chunker, embedding, and Qdrant storage.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import List, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    VlmPipelineOptions,
)
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.chunking import HybridChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from shared.config import settings, truncate_for_embedding
from shared.schemas import IngestionJobStatus, DocumentChunk
from shared.telemetry import get_tracer, IngestionMetrics

from .loader import (
    call_external_docling,
    fallback_pdf_extraction,
    get_file_format_map,
    get_pdf_metadata,
    load_json_file,
    load_text_file,
)
from .loader import _get_docling_prompt
from .chunker import create_chunks_from_content

logger = logging.getLogger(__name__)

# Global lock for thread-safe environment variable manipulation
_env_lock = threading.Lock()


class FileProcessor:
    """Processes files for ingestion."""

    def __init__(
        self,
        job_ctx,
        job_id: str,
        collection_name: str,
        processing_config: dict = None,
    ):
        self.job_ctx = job_ctx
        self.job_id = job_id
        self.collection_name = collection_name
        self.tracer = get_tracer()
        self.metrics = IngestionMetrics()

        self.embedding_client = OpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_API_BASE,
        )
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )

        self._converter = None
        self._chunker = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.embedding_model_id = (
            (processing_config or {}).get("embedding_model_id") or settings.EMBEDDING_MODEL
        )
        self.docling_model = (
            (processing_config or {}).get("docling_model") or settings.DOCLING_MODEL
        )

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.HTML,
                    InputFormat.MD,
                ],
            )
        return self._converter

    @property
    def chunker(self) -> Optional[HybridChunker]:
        if self._chunker is None and self.docling_model:
            try:
                # Use lock to ensure thread-safe environment variable manipulation
                # HybridChunker reads OPENAI_API_BASE/KEY from environment
                with _env_lock:
                    original_api_base = os.environ.get("OPENAI_API_BASE")
                    original_api_key = os.environ.get("OPENAI_API_KEY")
                    if settings.LLM_API_BASE:
                        os.environ["OPENAI_API_BASE"] = settings.LLM_API_BASE
                    if settings.LLM_API_KEY:
                        os.environ["OPENAI_API_KEY"] = settings.LLM_API_KEY
                    try:
                        self._chunker = HybridChunker(tokenizer=self.docling_model)
                    finally:
                        if original_api_base:
                            os.environ["OPENAI_API_BASE"] = original_api_base
                        elif "OPENAI_API_BASE" in os.environ and not original_api_base:
                            del os.environ["OPENAI_API_BASE"]
                        if original_api_key:
                            os.environ["OPENAI_API_KEY"] = original_api_key
                        elif "OPENAI_API_KEY" in os.environ and not original_api_key:
                            del os.environ["OPENAI_API_KEY"]
            except Exception as e:
                logger.warning("Failed to initialize HybridChunker: %s", e)
                return None
        return self._chunker

    async def process(
        self,
        file_path: str,
        original_filename: str,
        processing_config: dict,
    ) -> dict:
        with self.tracer.start_as_current_span("file_process") as span:
            span.set_attribute("filename", original_filename)
            span.set_attribute("file_path", file_path)
            start_time = time.time()

            # Validate file_path to prevent path traversal attacks
            from pathlib import Path
            try:
                upload_dir = Path(settings.UPLOAD_DIR).resolve()
                target_path = Path(file_path).resolve()

                # Use is_relative_to (safe against prefix collisions like
                # /upload_dir_evil matching /upload_dir)
                if not target_path.is_relative_to(upload_dir):
                    error_msg = f"Path traversal attempt detected: {file_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                file_path = str(target_path)
            except (ValueError, OSError) as e:
                logger.error("Invalid file path: %s - %s", file_path, e)
                raise ValueError(f"Invalid file path: {file_path}") from e

            await self._ensure_collection()

            file_ext = os.path.splitext(original_filename)[1].lower()
            span.set_attribute("file_type", file_ext)

            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EXTRACTING,
                progress=10,
                message=f"Extracting content from {original_filename}...",
            )

            extract_start = time.time()

            if file_ext == ".txt":
                content = load_text_file(file_path)
                await self.job_ctx.update_job_status(
                    self.job_id, IngestionJobStatus.PARTITIONING,
                    progress=40, message="Splitting content into chunks...",
                )
                partition_start = time.time()
                chunks = create_chunks_from_content(
                    content, original_filename, file_ext, self.job_id,
                    text_splitter=self.text_splitter,
                )
                partition_duration = (time.time() - partition_start) * 1000
            elif file_ext == ".json":
                content = load_json_file(file_path)
                await self.job_ctx.update_job_status(
                    self.job_id, IngestionJobStatus.PARTITIONING,
                    progress=40, message="Splitting content into chunks...",
                )
                partition_start = time.time()
                chunks = create_chunks_from_content(
                    content, original_filename, file_ext, self.job_id,
                    text_splitter=self.text_splitter,
                )
                partition_duration = (time.time() - partition_start) * 1000
            else:
                await self.job_ctx.update_job_status(
                    self.job_id, IngestionJobStatus.PARTITIONING,
                    progress=40, message="Extracting and chunking content with Docling...",
                )
                partition_start = time.time()
                chunks = await self._process_with_docling_loader(
                    file_path, original_filename, file_ext, processing_config
                )
                partition_duration = (time.time() - partition_start) * 1000

            extract_duration = (time.time() - extract_start) * 1000
            self.metrics.record_processing_time(extract_duration, "extraction")
            self.metrics.record_processing_time(partition_duration, "partition")
            span.set_attribute("chunks_created", len(chunks))

            if not chunks:
                logger.warning("No chunks created from %s", original_filename)
                self.metrics.record_extraction_failure("file", "no_chunks")
                return {"chunks_created": 0}

            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EMBEDDING,
                progress=70,
                message=f"Embedding {len(chunks)} chunks...",
            )

            embed_start = time.time()
            await self._embed_and_store(chunks)
            embed_duration = (time.time() - embed_start) * 1000

            self.metrics.record_processing_time(embed_duration, "embedding")
            self.metrics.record_chunks_created(len(chunks), self.collection_name)
            self.metrics.record_document_processed(self.collection_name, "file")

            return {
                "chunks_created": len(chunks),
                "extraction_duration_ms": extract_duration,
                "partition_duration_ms": partition_duration,
                "embedding_duration_ms": embed_duration,
                "total_duration_ms": (time.time() - start_time) * 1000,
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

    async def _process_with_docling_loader(
        self,
        file_path: str,
        original_filename: str,
        file_ext: str,
        processing_config: dict,
    ) -> List[DocumentChunk]:
        content = await call_external_docling(file_path, original_filename)
        if content:
            logger.info("Using external Docling API for %s", original_filename)
            return create_chunks_from_content(
                content, original_filename, file_ext, self.job_id,
                text_splitter=self.text_splitter,
            )

        try:
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                self._run_docling_loader,
                file_path,
                original_filename,
            )
            return chunks
        except Exception as e:
            logger.error("DoclingLoader processing failed: %s", e)
            self.metrics.record_extraction_failure("file", str(type(e).__name__))
            if file_ext == ".pdf":
                content = fallback_pdf_extraction(file_path)
                if content:
                    return create_chunks_from_content(
                        content, original_filename, file_ext, self.job_id,
                        text_splitter=self.text_splitter,
                    )
            raise

    def _run_docling_loader(
        self, file_path: str, original_filename: str
    ) -> List[DocumentChunk]:
        file_ext = os.path.splitext(original_filename)[1].lower()
        fmt_map = get_file_format_map()
        input_format = fmt_map.get(file_ext)

        if input_format != InputFormat.PDF:
            result = self.converter.convert(file_path)
        else:
            try:
                from docling.datamodel.pipeline_options import ApiVlmOptions
                pipeline_options = VlmPipelineOptions(enable_remote_services=True)
                vlm_url = (
                    getattr(settings, "VLM_API_BASE", None) or settings.LLM_API_BASE
                ).rstrip("/")
                vlm_model = getattr(settings, "VLM_MODEL", settings.DOCLING_MODEL)
                vlm_api_key = getattr(settings, "VLM_API_KEY", settings.LLM_API_KEY)
                vlm_options_params = {
                    "model": vlm_model,
                    "max_tokens": 4096,
                    "skip_special_tokens": False,
                }
                try:
                    vlm_options = ApiVlmOptions(
                        url=vlm_url,
                        params=vlm_options_params,
                        prompt=_get_docling_prompt(),
                        timeout=60.0,
                        concurrency=1,
                        temperature=0.0,
                        scale=2.0,
                    )
                except TypeError:
                    vlm_options = ApiVlmOptions(
                        url=vlm_url,
                        params=vlm_options_params,
                        prompt=_get_docling_prompt(),
                        timeout=60.0,
                        concurrency=1,
                        temperature=0.0,
                    )
                pipeline_options.vlm_options = vlm_options
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline,
                            pipeline_options=pipeline_options,
                        )
                    }
                )
                result = converter.convert(source=str(file_path))
            except (ImportError, AttributeError, Exception) as e:
                logger.warning("VlmPipeline not available (%s), using standard PDF", e)
                result = self.converter.convert(file_path)

        file_type = os.path.splitext(original_filename)[1].lstrip(".")
        doc_meta = {"source_type": "file", "file_type": file_type}
        if file_ext == ".pdf":
            doc_meta.update(get_pdf_metadata(file_path))

        doc = result.document
        chunker = self.chunker
        if chunker is None:
            try:
                from docling.chunking import HierarchicalChunker
                chunker = HierarchicalChunker()
            except ImportError:
                chunker = None

        if chunker is not None:
            chunk_iter = chunker.chunk(dl_doc=doc)
            doc_chunks = list(chunk_iter)
            chunks = []
            for i, chunk_data in enumerate(doc_chunks):
                page_content = chunker.contextualize(chunk=chunk_data)
                page_no = None
                prov = getattr(chunk_data, "prov", None)
                if prov and len(prov) > 0 and getattr(prov[0], "page_no", None) is not None:
                    page_no = getattr(prov[0], "page_no", None)
                meta = {**doc_meta}
                if page_no is not None:
                    meta["page_number"] = page_no
                chunks.append(
                    DocumentChunk(
                        id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                        content=page_content,
                        metadata=meta,
                        source_file=original_filename,
                        chunk_index=i,
                        total_chunks=len(doc_chunks),
                    )
                )
        else:
            markdown_text = getattr(doc, "export_to_markdown", lambda: None)()
            if not markdown_text:
                raise RuntimeError("DoclingDocument has no export_to_markdown")
            parts = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
            chunks = [
                DocumentChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=page_content,
                    metadata={**doc_meta},
                    source_file=original_filename,
                    chunk_index=i,
                    total_chunks=len(parts),
                )
                for i, page_content in enumerate(parts)
            ]

        logger.info("Created %d chunks from %s using docling", len(chunks), original_filename)
        return chunks

    async def extract_and_chunk(
        self,
        file_path: str,
        original_filename: str,
        processing_config: dict | None = None,
    ) -> List[DocumentChunk]:
        """Extract and chunk only (no embedding/storage). For CLI use."""
        file_ext = os.path.splitext(original_filename)[1].lower()
        processing_config = processing_config or {}

        if file_ext == ".txt":
            content = load_text_file(file_path)
            return create_chunks_from_content(
                content, original_filename, file_ext, self.job_id,
                text_splitter=self.text_splitter,
            )
        if file_ext == ".json":
            content = load_json_file(file_path)
            return create_chunks_from_content(
                content, original_filename, file_ext, self.job_id,
                text_splitter=self.text_splitter,
            )
        return await self._process_with_docling_loader(
            file_path, original_filename, file_ext, processing_config,
        )

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
