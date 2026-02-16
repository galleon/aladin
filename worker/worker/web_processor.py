"""
Web content processor using crawl4ai.
"""

import asyncio
import logging
import time
import uuid
import re
from typing import Any, List, Set
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings, truncate_for_embedding
from shared.schemas import IngestionJobStatus, DocumentChunk, CrawlStrategy
from shared.telemetry import get_tracer, IngestionMetrics

logger = logging.getLogger(__name__)


class WebProcessor:
    """Processes web content for ingestion."""

    def __init__(self, job_ctx, job_id: str, collection_name: str):
        self.job_ctx = job_ctx
        self.job_id = job_id
        self.collection_name = collection_name
        self.tracer = get_tracer()
        self.metrics = IngestionMetrics()

        # Initialize clients
        self.embedding_client = OpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_API_BASE,
        )

        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )

        # Text splitter
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
        """
        Process a web URL and its linked pages.

        Returns:
            dict with processing results
        """
        with self.tracer.start_as_current_span("web_process") as span:
            span.set_attribute("url", url)

            start_time = time.time()

            # Ensure collection exists
            await self._ensure_collection()

            # Update status
            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.CRAWLING,
                progress=10,
                message="Starting web crawl...",
            )

            # Crawl pages
            crawl_start = time.time()
            pages = await self._crawl_pages(url, source_config, processing_config)
            crawl_duration = (time.time() - crawl_start) * 1000

            self.metrics.record_processing_time(crawl_duration, "crawl")
            span.set_attribute("pages_crawled", len(pages))

            # Update status
            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.PARTITIONING,
                progress=40,
                message=f"Processing {len(pages)} pages...",
                pages_total=len(pages),
            )

            # Process and chunk content
            partition_start = time.time()
            all_chunks = []

            for i, page in enumerate(pages):
                chunks = await self._process_page(page, metadata)
                all_chunks.extend(chunks)

                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.PARTITIONING,
                    progress=40 + int(30 * (i + 1) / len(pages)),
                    pages_processed=i + 1,
                )

            partition_duration = (time.time() - partition_start) * 1000
            self.metrics.record_processing_time(partition_duration, "partition")

            if not all_chunks:
                logger.warning(f"No content extracted from {url}")
                return {"chunks_created": 0, "pages_crawled": len(pages)}

            # Update status
            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EMBEDDING,
                progress=75,
                message=f"Embedding {len(all_chunks)} chunks...",
            )

            # Embed and store
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
        """Ensure Qdrant collection exists."""
        collections = self.qdrant.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            # Dynamically detect embedding dimension by testing the model
            try:
                test_embedding = self.embedding_client.embeddings.create(
                    input=["test"],
                    model=self.embedding_model_id,
                )
                if test_embedding.data and len(test_embedding.data) > 0:
                    vector_size = len(test_embedding.data[0].embedding)
                    logger.info(
                        f"Detected embedding dimension: {vector_size} for model: {self.embedding_model_id}"
                    )
                else:
                    vector_size = settings.EMBEDDING_DIMENSION
                    logger.warning(
                        f"Could not detect embedding dimension, using default: {vector_size}"
                    )
            except Exception as e:
                vector_size = settings.EMBEDDING_DIMENSION
                logger.warning(
                    f"Failed to detect embedding dimension ({e}), using default: {vector_size}"
                )

            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"Created collection: {self.collection_name} with dimension: {vector_size}"
            )

    async def _crawl_pages(
        self,
        start_url: str,
        source_config: dict,
        processing_config: dict,
    ) -> List[dict]:
        """
        Crawl web pages using crawl4ai.

        Returns:
            List of page data dictionaries
        """
        pages = []
        visited: Set[str] = set()
        queue = [(start_url, 0)]  # (url, depth)

        depth_limit = source_config.get("depth_limit", settings.DEFAULT_CRAWL_DEPTH)
        max_pages = source_config.get("max_pages", 100)
        strategy = source_config.get("strategy", "bfs")
        inclusion_patterns = source_config.get("inclusion_patterns", [])
        exclusion_patterns = source_config.get("exclusion_patterns", [])

        # Browser configuration
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        # Crawler configuration
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_for=processing_config.get("wait_for_selector"),
            page_timeout=processing_config.get("wait_timeout", 30) * 1000,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            while queue and len(pages) < max_pages:
                if strategy == "bfs":
                    url, depth = queue.pop(0)
                else:  # DFS
                    url, depth = queue.pop()

                # Normalize URL
                url = url.rstrip("/")

                if url in visited:
                    continue

                if depth > depth_limit:
                    continue

                # Check patterns
                if not self._matches_patterns(
                    url, inclusion_patterns, exclusion_patterns
                ):
                    continue

                visited.add(url)

                try:
                    logger.info(f"Crawling: {url} (depth {depth})")

                    result = await crawler.arun(url=url, config=crawler_config)

                    if result.success:
                        page_data = {
                            "url": url,
                            "depth": depth,
                            "title": result.metadata.get("title", ""),
                            "markdown": result.markdown,
                            "html": result.html
                            if processing_config.get("extract_tables")
                            else None,
                            "links": result.links.get("internal", [])
                            if hasattr(result, "links")
                            else [],
                        }

                        # Extract tables if configured
                        if processing_config.get("extract_tables"):
                            page_data["tables"] = self._extract_tables(
                                result.html,
                                processing_config.get("table_strategy", "markdown"),
                            )

                        pages.append(page_data)
                        self.metrics.pages_crawled.add(
                            1, {"collection": self.collection_name}
                        )

                        # Add links to queue
                        if depth < depth_limit:
                            for link in page_data.get("links", []):
                                link_url = (
                                    link.get("href", "")
                                    if isinstance(link, dict)
                                    else str(link)
                                )
                                if link_url:
                                    absolute_url = urljoin(url, link_url)
                                    if absolute_url not in visited:
                                        queue.append((absolute_url, depth + 1))
                    else:
                        logger.warning(f"Failed to crawl {url}: {result.error_message}")
                        self.metrics.record_extraction_failure("web", "crawl_failed")

                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    self.metrics.record_extraction_failure("web", str(type(e).__name__))

        return pages

    def _matches_patterns(
        self,
        url: str,
        inclusion_patterns: List[str],
        exclusion_patterns: List[str],
    ) -> bool:
        """Check if URL matches inclusion/exclusion patterns."""
        # Check exclusion first
        for pattern in exclusion_patterns:
            if self._url_matches_pattern(url, pattern):
                return False

        # If no inclusion patterns, accept all
        if not inclusion_patterns:
            return True

        # Check inclusion
        for pattern in inclusion_patterns:
            if self._url_matches_pattern(url, pattern):
                return True

        return False

    def _url_matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if URL matches a glob-like pattern."""
        # Convert glob pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.search(regex_pattern, url))

    def _extract_tables(self, html: str, strategy: str) -> List[dict]:
        """Extract tables from HTML content."""
        tables = []

        if not html:
            return tables

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            for i, table in enumerate(soup.find_all("table")):
                table_data = {
                    "index": i,
                    "html": str(table),
                }

                if strategy == "markdown":
                    # Convert to markdown
                    table_data["markdown"] = self._table_to_markdown(table)
                elif strategy == "summary_and_html":
                    # Keep HTML and add placeholder for summary
                    table_data["markdown"] = self._table_to_markdown(table)
                    table_data["needs_summary"] = True

                tables.append(table_data)

        except Exception as e:
            logger.warning(f"Failed to extract tables: {e}")

        return tables

    def _table_to_markdown(self, table) -> str:
        """Convert HTML table to markdown."""
        rows = []

        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append("| " + " | ".join(cells) + " |")

        if len(rows) > 1:
            # Add separator after header
            header_len = rows[0].count("|") - 1
            separator = "|" + "|".join(["---"] * header_len) + "|"
            rows.insert(1, separator)

        return "\n".join(rows)

    async def _process_page(self, page: dict, metadata: dict) -> List[DocumentChunk]:
        """Process a single page into chunks."""
        chunks = []

        content = page.get("markdown", "")
        if not content:
            return chunks

        # Add tables to content
        for table in page.get("tables", []):
            if table.get("markdown"):
                content += f"\n\n{table['markdown']}"

        # Split into chunks
        texts = self.text_splitter.split_text(content)

        for i, text in enumerate(texts):
            chunk = DocumentChunk(
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
            chunks.append(chunk)

        return chunks

    async def _embed_and_store(self, chunks: List[DocumentChunk]):
        """Embed chunks and store in Qdrant."""
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Truncate to fit embedding model context (e.g. 512 tokens)
            texts = [truncate_for_embedding(c.content) for c in batch]

            response = self.embedding_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )

            embeddings = [e.embedding for e in response.data]

            # Create points (Qdrant accepts only UUID or int; use deterministic UUID from chunk.id)
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

            # Upsert to Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Stored {len(points)} chunks in {self.collection_name}")
