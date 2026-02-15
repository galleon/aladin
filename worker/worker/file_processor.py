"""
File content processor using docling.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, List, Optional

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

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings, truncate_for_embedding
from shared.schemas import IngestionJobStatus, DocumentChunk
from shared.telemetry import get_tracer, IngestionMetrics

from .prompts_loader import get_prompt as get_config_prompt

logger = logging.getLogger(__name__)


def _get_docling_prompt() -> str:
    """Get docling conversion prompt from config or default."""
    return get_config_prompt("file.docling_convert") or "Convert this page to docling."


# File extension to input format mapping
FILE_FORMAT_MAP = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".doc": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
    ".ppt": InputFormat.PPTX,
    ".html": InputFormat.HTML,
    ".htm": InputFormat.HTML,
    ".md": InputFormat.MD,
    ".txt": None,  # Plain text, handle separately
    ".json": None,  # JSON, handle separately
}


class FileProcessor:
    """Processes files for ingestion."""

    def __init__(
        self, job_ctx, job_id: str, collection_name: str, processing_config: dict = None
    ):
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

        # Initialize docling converter and chunker
        self._converter = None
        self._chunker = None

        # Text splitters for fallback when Docling fails (e.g. HF 502) or for non-PDF content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._markdown_splitter = None

        # Get embedding model ID from processing config or settings
        self.embedding_model_id = None
        if processing_config:
            self.embedding_model_id = processing_config.get("embedding_model_id")
        if not self.embedding_model_id:
            self.embedding_model_id = settings.EMBEDDING_MODEL

        # Get docling model for HybridChunker (defaults to granite-docling-258M)
        self.docling_model = None
        if processing_config:
            self.docling_model = processing_config.get("docling_model")
        if not self.docling_model:
            self.docling_model = settings.DOCLING_MODEL

    @property
    def converter(self) -> DocumentConverter:
        """Lazy-load docling converter."""
        if self._converter is None:
            # Configure PDF pipeline
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
        """
        Lazy-load HybridChunker with docling model from LLM_API_BASE.

        The model (e.g., ibm-granite/granite-docling-258M) should be available
        on the LLM_API_BASE endpoint. The HybridChunker will use this model
        for tokenization-aware chunking.
        """
        if self._chunker is None and self.docling_model:
            try:
                # Configure environment for custom API base
                # Models available on LLM_API_BASE (OpenAI-compatible endpoint)
                # can be accessed via OPENAI_API_BASE environment variable
                import os

                original_api_base = os.environ.get("OPENAI_API_BASE")
                original_api_key = os.environ.get("OPENAI_API_KEY")

                # Set API base for OpenAI-compatible endpoints
                # This allows HybridChunker to access models from LLM_API_BASE
                if settings.LLM_API_BASE:
                    os.environ["OPENAI_API_BASE"] = settings.LLM_API_BASE
                if settings.LLM_API_KEY:
                    os.environ["OPENAI_API_KEY"] = settings.LLM_API_KEY

                try:
                    # Initialize HybridChunker with docling model
                    # Model format: "ibm-granite/granite-docling-258M"
                    # This will use the model from LLM_API_BASE for tokenization
                    self._chunker = HybridChunker(tokenizer=self.docling_model)
                    logger.info(
                        f"Initialized HybridChunker with model: {self.docling_model} "
                        f"(API base: {settings.LLM_API_BASE})"
                    )
                finally:
                    # Restore original environment
                    if original_api_base:
                        os.environ["OPENAI_API_BASE"] = original_api_base
                    elif "OPENAI_API_BASE" in os.environ and not original_api_base:
                        del os.environ["OPENAI_API_BASE"]
                    if original_api_key:
                        os.environ["OPENAI_API_KEY"] = original_api_key
                    elif "OPENAI_API_KEY" in os.environ and not original_api_key:
                        del os.environ["OPENAI_API_KEY"]

            except Exception as e:
                logger.warning(
                    f"Failed to initialize HybridChunker: {e}. Continuing without chunker."
                )
                return None
        return self._chunker

    async def process(
        self,
        file_path: str,
        original_filename: str,
        processing_config: dict,
    ) -> dict:
        """
        Process a file for ingestion.

        Returns:
            dict with processing results
        """
        with self.tracer.start_as_current_span("file_process") as span:
            span.set_attribute("filename", original_filename)
            span.set_attribute("file_path", file_path)

            start_time = time.time()

            # Ensure collection exists
            await self._ensure_collection()

            # Determine file type
            file_ext = os.path.splitext(original_filename)[1].lower()
            span.set_attribute("file_type", file_ext)

            # Update status
            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EXTRACTING,
                progress=10,
                message=f"Extracting content from {original_filename}...",
            )

            # Extract content and create chunks using DoclingLoader
            extract_start = time.time()

            if file_ext in [".txt"]:
                # For plain text, process directly
                content = await self._process_text_file(file_path)
                # Update status
                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.PARTITIONING,
                    progress=40,
                    message="Splitting content into chunks...",
                )
                # Split into chunks
                partition_start = time.time()
                chunks = await self._create_chunks(content, original_filename, file_ext)
                partition_duration = (time.time() - partition_start) * 1000
            elif file_ext in [".json"]:
                # For JSON, process directly
                content = await self._process_json_file(file_path)
                # Update status
                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.PARTITIONING,
                    progress=40,
                    message="Splitting content into chunks...",
                )
                # Split into chunks
                partition_start = time.time()
                chunks = await self._create_chunks(content, original_filename, file_ext)
                partition_duration = (time.time() - partition_start) * 1000
            else:
                # Use DoclingLoader with ExportType.DOC_CHUNKS
                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.PARTITIONING,
                    progress=40,
                    message="Extracting and chunking content with Docling...",
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
                logger.warning(f"No chunks created from {original_filename}")
                self.metrics.record_extraction_failure("file", "no_chunks")
                return {"chunks_created": 0}

            if not chunks:
                return {"chunks_created": 0}

            # Update status
            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EMBEDDING,
                progress=70,
                message=f"Embedding {len(chunks)} chunks...",
            )

            # Embed and store
            embed_start = time.time()
            await self._embed_and_store(chunks)
            embed_duration = (time.time() - embed_start) * 1000

            self.metrics.record_processing_time(embed_duration, "embedding")
            self.metrics.record_chunks_created(len(chunks), self.collection_name)
            self.metrics.record_document_processed(self.collection_name, "file")

            total_duration = (time.time() - start_time) * 1000

            return {
                "chunks_created": len(chunks),
                "extraction_duration_ms": extract_duration,
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

    async def _process_text_file(self, file_path: str) -> str:
        """Process a plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    async def _process_json_file(self, file_path: str) -> str:
        """Process a JSON file by converting to readable text."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert JSON to readable format
        return self._json_to_text(data)

    def _json_to_text(self, data, indent: int = 0) -> str:
        """Convert JSON data to readable text."""
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._json_to_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)

    async def _call_external_docling(
        self, file_path: str, original_filename: str
    ) -> Optional[str]:
        """
        Call external Docling at DOCLING_API_BASE / LLM_API_BASE.
        1) POST multipart to /documents/convert or /convert (docling-api style).
        2) If that fails, call OpenAI-compatible /chat/completions with model DOCLING_MODEL.
        Returns markdown string or None on failure.
        """
        base = (settings.DOCLING_API_BASE or "").strip().rstrip("/")
        if not base:
            return None
        import base64
        import httpx

        # 1) Try docling-api style: POST /documents/convert
        try:
            url = f"{base}/documents/convert"
            with open(file_path, "rb") as f:
                files = {"document": (original_filename, f)}
                async with httpx.AsyncClient(timeout=600.0) as client:
                    r = await client.post(url, files=files)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "application/json" in ct:
                data = r.json()
                out = (
                    data.get("markdown")
                    or data.get("text")
                    or data.get("content")
                    or (
                        data.get("data", {}).get("markdown")
                        if isinstance(data.get("data"), dict)
                        else None
                    )
                )
                if out:
                    return out
            return r.text or None
        except Exception as e:
            logger.debug(f"External Docling /documents/convert failed: {e}")

        # 2) Try /convert
        try:
            url2 = f"{base}/convert"
            with open(file_path, "rb") as f:
                files = {"document": (original_filename, f)}
                async with httpx.AsyncClient(timeout=600.0) as client:
                    r2 = await client.post(url2, files=files)
            r2.raise_for_status()
            ct = r2.headers.get("content-type", "")
            if "application/json" in ct:
                data = r2.json()
                out = (
                    data.get("markdown")
                    or data.get("text")
                    or data.get("content")
                    or None
                )
                if out:
                    return out
            return r2.text or None
        except Exception as e2:
            logger.debug(f"External Docling /convert failed: {e2}")

        # 3) OpenAI-compatible API: /chat/completions with model granite_docling (document as base64)
        model = settings.DOCLING_MODEL
        try:
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            file_ext = os.path.splitext(original_filename)[1].lower()
            mime = (
                "application/pdf" if file_ext == ".pdf" else "application/octet-stream"
            )

            # Some servers accept document in user content (e.g. input_document or image_url-style data URL)
            data_url = f"data:{mime};base64,{b64}"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Convert this document to markdown.",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "max_tokens": 128 * 1024,
            }
            async with httpx.AsyncClient(timeout=600.0) as client:
                r3 = await client.post(
                    f"{base}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            r3.raise_for_status()
            data = r3.json()
            choice = (data.get("choices") or [None])[0]
            if choice and isinstance(choice.get("message"), dict):
                content = choice["message"].get("content")
                if isinstance(content, str) and content.strip():
                    return content
            return None
        except Exception as e3:
            logger.warning(f"External Docling chat/completions ({model}) failed: {e3}")
        return None

    async def _process_with_docling_loader(
        self,
        file_path: str,
        original_filename: str,
        file_ext: str,
        processing_config: dict,
    ) -> List[DocumentChunk]:
        """Process document: try external Docling API first (if DOCLING_API_BASE/LLM_API_BASE), else in-process docling."""
        # Prefer external Docling when DOCLING_API_BASE (or LLM_API_BASE) is set
        content = await self._call_external_docling(file_path, original_filename)
        if content:
            logger.info(f"Using external Docling API for {original_filename}")
            return await self._create_chunks(content, original_filename, file_ext)

        try:
            # Run in-process docling in thread pool (synchronous)
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                self._run_docling_loader,
                file_path,
                original_filename,
            )

            return chunks

        except Exception as e:
            logger.error(f"DoclingLoader processing failed: {e}")
            self.metrics.record_extraction_failure("file", str(type(e).__name__))

            # Try fallback for PDFs
            if file_ext == ".pdf":
                content = await self._fallback_pdf_extraction(file_path)
                if content:
                    return await self._create_chunks(
                        content, original_filename, file_ext
                    )

            raise

    def _run_docling_loader(
        self,
        file_path: str,
        original_filename: str,
    ) -> List[DocumentChunk]:
        """Process document using docling directly with VlmPipeline."""
        import os

        # Determine file format
        file_ext = os.path.splitext(original_filename)[1].lower()
        input_format = FILE_FORMAT_MAP.get(file_ext)

        if input_format != InputFormat.PDF:
            # For non-PDF files, use standard converter
            result = self.converter.convert(file_path)
        else:
            # For PDFs, try to use VlmPipeline with custom API configuration
            try:
                from docling.datamodel.pipeline_options import ApiVlmOptions

                # Configure VLM pipeline options
                pipeline_options = VlmPipelineOptions(enable_remote_services=True)

                # Get VLM configuration from settings
                vlm_url = (
                    settings.VLM_API_BASE.rstrip("/")
                    if hasattr(settings, "VLM_API_BASE")
                    else settings.LLM_API_BASE.rstrip("/")
                )
                vlm_model = (
                    settings.VLM_MODEL
                    if hasattr(settings, "VLM_MODEL")
                    else settings.DOCLING_MODEL
                )
                vlm_api_key = (
                    settings.VLM_API_KEY
                    if hasattr(settings, "VLM_API_KEY")
                    else settings.LLM_API_KEY
                )

                # Configure ApiVlmOptions
                vlm_options_params = {
                    "model": vlm_model,
                    "max_tokens": 4096,
                    "skip_special_tokens": False,
                }

                # Try to set scale if supported
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
                    # scale might not be supported in this version
                    vlm_options = ApiVlmOptions(
                        url=vlm_url,
                        params=vlm_options_params,
                        prompt=_get_docling_prompt(),
                        timeout=60.0,
                        concurrency=1,
                        temperature=0.0,
                    )

                pipeline_options.vlm_options = vlm_options

                # Create converter with VlmPipeline for PDFs
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
                # Fallback if VlmPipeline or ApiVlmOptions is not available
                logger.warning(
                    f"VlmPipeline not available ({e}), using standard PDF pipeline"
                )
                result = self.converter.convert(file_path)

        # Document-level metadata (PDF: author, title, dates when available)
        file_type = os.path.splitext(original_filename)[1].lstrip(".")
        doc_meta: dict = {
            "source_type": "file",
            "file_type": file_type,
        }
        if file_ext == ".pdf":
            doc_meta.update(self._get_pdf_metadata(file_path))

        # Chunk document using docling 2.x API: chunker.chunk(dl_doc) + chunker.contextualize(chunk)
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
                chunk = DocumentChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=page_content,
                    metadata=meta,
                    source_file=original_filename,
                    chunk_index=i,
                    total_chunks=len(doc_chunks),
                )
                chunks.append(chunk)
        else:
            # Fallback: export to markdown and split into chunks
            markdown_text = getattr(doc, "export_to_markdown", lambda: None)()
            if not markdown_text:
                raise RuntimeError(
                    "DoclingDocument has no export_to_markdown and no chunker available"
                )
            parts = [p.strip() for p in markdown_text.split("\n\n") if p.strip()]
            chunks = []
            for i, page_content in enumerate(parts):
                chunk = DocumentChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=page_content,
                    metadata={**doc_meta},
                    source_file=original_filename,
                    chunk_index=i,
                    total_chunks=len(parts),
                )
                chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from {original_filename} using docling converter"
        )
        return chunks

    def _get_pdf_metadata(self, file_path: str) -> dict:
        """Extract document-level metadata from a PDF (author, title, dates). Returns only non-empty keys."""
        out: dict = {}
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            raw = reader.metadata
            if not raw:
                return out
            if raw.get("/Author"):
                out["author"] = str(raw["/Author"]).strip()
            if raw.get("/Title"):
                out["title"] = str(raw["/Title"]).strip()
            if raw.get("/CreationDate"):
                out["creation_date"] = str(raw["/CreationDate"]).strip()
            if raw.get("/ModDate"):
                out["modification_date"] = str(raw["/ModDate"]).strip()
        except Exception as e:
            logger.debug(f"Could not read PDF metadata from {file_path}: {e}")
        return out

    async def _fallback_pdf_extraction(self, file_path: str) -> str:
        """Fallback PDF extraction using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            pages = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            return "\n\n".join(pages)

        except Exception as e:
            logger.error(f"Fallback PDF extraction failed: {e}")
            return ""

    async def _create_chunks(
        self,
        content: str,
        filename: str,
        file_ext: str,
    ) -> List[DocumentChunk]:
        """Create document chunks from content."""
        chunks = []

        # For markdown, try to split by headers first (if markdown_splitter is available)
        markdown_splitter = getattr(self, "_markdown_splitter", None)
        if markdown_splitter and (file_ext in [".md"] or content.startswith("#")):
            try:
                header_splits = markdown_splitter.split_text(content)

                for doc in header_splits:
                    # Further split long sections
                    section_texts = self.text_splitter.split_text(doc.page_content)

                    for i, text in enumerate(section_texts):
                        chunk = DocumentChunk(
                            id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                            content=text,
                            metadata={
                                "source_type": "file",
                                "file_type": file_ext.lstrip("."),
                                **doc.metadata,
                            },
                            source_file=filename,
                            chunk_index=len(chunks),
                            total_chunks=0,  # Will update after
                        )
                        chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Markdown splitting failed, using regular: {e}")
                chunks = []

        # Regular splitting if markdown failed or not markdown
        if not chunks:
            texts = self.text_splitter.split_text(content)

            for i, text in enumerate(texts):
                chunk = DocumentChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=text,
                    metadata={
                        "source_type": "file",
                        "file_type": file_ext.lstrip("."),
                    },
                    source_file=filename,
                    chunk_index=i,
                    total_chunks=len(texts),
                )
                chunks.append(chunk)

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    async def _embed_and_store(self, chunks: List[DocumentChunk]):
        """Embed chunks and store in Qdrant."""
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Truncate texts to fit embedding model context window (e.g. 512 tokens)
            texts = [truncate_for_embedding(c.content) for c in batch]

            response = self.embedding_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )

            embeddings = [e.embedding for e in response.data]

            # Create points (Qdrant accepts only UUID or int; chunk.id is a string, so use deterministic UUID)
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
