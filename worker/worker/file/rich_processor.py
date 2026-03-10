"""
Rich file processor: nv-ingest-style pipeline using local models only.

Compared to the default FileProcessor (Docling text-only), this processor:
  - Stores TEXT, TABLE (structured), and IMAGE chunks as separate Qdrant points
  - Extracts bounding boxes per chunk from Docling provenance data
  - Captions embedded images using the VLM (VLM_API_BASE, e.g. cosmos-reason2-8b)
  - Stores NeMo-Retriever-compatible metadata per chunk:
      content_type, text_type, text_location [l,t,r,b], page_width, page_height,
      language (placeholder), table_content, table_format, image_caption

No external nv-ingest service required — uses Docling + pypdfium2 + vLLM already
running on DGX.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
import uuid
from typing import List, Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.config import settings, truncate_for_embedding
from shared.schemas import DocumentChunk, IngestionJobStatus
from shared.telemetry import IngestionMetrics, get_tracer

from .loader import (
    _get_docling_prompt,
    call_external_docling,
    fallback_pdf_extraction,
    get_file_format_map,
    get_pdf_metadata,
    load_json_file,
    load_text_file,
)
from .chunker import create_chunks_from_content

logger = logging.getLogger(__name__)

_env_lock = threading.Lock()

# Content type constants (matches NeMo-Retriever MetadataSchema)
CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_STRUCTURED = "structured"  # tables / charts
CONTENT_TYPE_IMAGE = "image"


class RichChunk:
    """Extended chunk with NeMo-style metadata (bbox, content_type, etc.)."""

    __slots__ = (
        "id", "content", "source_file", "chunk_index", "total_chunks",
        "page_number", "page_width", "page_height",
        "content_type",   # text | structured | image
        "text_type",      # body | header | footnote | caption | list_item | table | picture
        "text_location",  # [l, t, r, b] in page points (Docling coord space)
        "table_content",  # raw table markdown (structured only)
        "table_format",   # "markdown" (structured only)
        "image_caption",  # VLM caption (image only)
        "metadata",       # extra flat metadata (source_type, file_type, pdf meta, …)
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))
        if self.metadata is None:
            self.metadata = {}


class RichFileProcessor:
    """
    nv-ingest-style file processor using local models.

    Switching from the default FileProcessor:
      - Set FILE_PROCESSOR=rich in .env (global default)
      - Or pass {"file_processor": "rich"} in processing_config per job
    """

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
        self.vlm_client = OpenAI(
            api_key=getattr(settings, "VLM_API_KEY", settings.LLM_API_KEY),
            base_url=getattr(settings, "VLM_API_BASE", settings.LLM_API_BASE),
        )
        self.qdrant = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )

        self.processing_config = processing_config or {}
        self.embedding_model_id = (
            self.processing_config.get("embedding_model_id") or settings.EMBEDDING_MODEL
        )
        self.docling_model = (
            self.processing_config.get("docling_model") or settings.DOCLING_MODEL
        )
        self.vlm_model = getattr(settings, "VLM_MODEL", "cosmos-reason2-8b")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._converter = None
        self._chunker = None

    # ------------------------------------------------------------------
    # Docling helpers (mirrors FileProcessor to reuse logic)
    # ------------------------------------------------------------------

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX,
                    InputFormat.HTML, InputFormat.MD,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=PdfPipelineOptions(
                            do_table_structure=True,
                            table_structure_options=type(
                                "T", (), {"mode": TableFormerMode.ACCURATE}
                            )(),
                        )
                    )
                },
            )
        return self._converter

    @property
    def chunker(self) -> Optional[HybridChunker]:
        if self._chunker is None and self.docling_model:
            try:
                with _env_lock:
                    orig_base = os.environ.get("OPENAI_API_BASE")
                    orig_key = os.environ.get("OPENAI_API_KEY")
                    if settings.LLM_API_BASE:
                        os.environ["OPENAI_API_BASE"] = settings.LLM_API_BASE
                    if settings.LLM_API_KEY:
                        os.environ["OPENAI_API_KEY"] = settings.LLM_API_KEY
                    try:
                        self._chunker = HybridChunker(tokenizer=self.docling_model)
                    finally:
                        if orig_base:
                            os.environ["OPENAI_API_BASE"] = orig_base
                        elif "OPENAI_API_BASE" in os.environ:
                            del os.environ["OPENAI_API_BASE"]
                        if orig_key:
                            os.environ["OPENAI_API_KEY"] = orig_key
                        elif "OPENAI_API_KEY" in os.environ:
                            del os.environ["OPENAI_API_KEY"]
            except Exception as e:
                logger.warning("HybridChunker init failed: %s", e)
        return self._chunker

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process(
        self,
        file_path: str,
        original_filename: str,
        processing_config: dict,
    ) -> dict:
        with self.tracer.start_as_current_span("rich_file_process") as span:
            span.set_attribute("filename", original_filename)
            start_time = time.time()

            from pathlib import Path
            try:
                upload_dir = Path(settings.UPLOAD_DIR).resolve()
                target_path = Path(file_path).resolve()
                if not target_path.is_relative_to(upload_dir):
                    raise ValueError(f"Path traversal attempt: {file_path}")
                file_path = str(target_path)
            except (ValueError, OSError) as e:
                raise ValueError(f"Invalid file path: {file_path}") from e

            await self._ensure_collection()

            file_ext = os.path.splitext(original_filename)[1].lower()

            await self.job_ctx.update_job_status(
                self.job_id, IngestionJobStatus.EXTRACTING, progress=10,
                message=f"Extracting {original_filename}...",
            )

            extract_start = time.time()

            if file_ext == ".txt":
                content = load_text_file(file_path)
                plain_chunks = create_chunks_from_content(
                    content, original_filename, file_ext, self.job_id,
                    text_splitter=self.text_splitter,
                )
                rich_chunks = [self._plain_to_rich(c) for c in plain_chunks]
            elif file_ext == ".json":
                content = load_json_file(file_path)
                plain_chunks = create_chunks_from_content(
                    content, original_filename, file_ext, self.job_id,
                    text_splitter=self.text_splitter,
                )
                rich_chunks = [self._plain_to_rich(c) for c in plain_chunks]
            else:
                await self.job_ctx.update_job_status(
                    self.job_id, IngestionJobStatus.PARTITIONING, progress=30,
                    message="Extracting structure, tables, and images...",
                )
                rich_chunks = await self._process_rich(
                    file_path, original_filename, file_ext,
                )

            extract_duration = (time.time() - extract_start) * 1000
            span.set_attribute("chunks_created", len(rich_chunks))

            if not rich_chunks:
                logger.warning("No chunks from %s", original_filename)
                return {"chunks_created": 0}

            await self.job_ctx.update_job_status(
                self.job_id, IngestionJobStatus.EMBEDDING, progress=70,
                message=f"Embedding {len(rich_chunks)} chunks ({self._type_summary(rich_chunks)})...",
            )

            embed_start = time.time()
            await self._embed_and_store(rich_chunks)
            embed_duration = (time.time() - embed_start) * 1000

            self.metrics.record_chunks_created(len(rich_chunks), self.collection_name)
            self.metrics.record_document_processed(self.collection_name, "file")

            return {
                "chunks_created": len(rich_chunks),
                "extraction_duration_ms": extract_duration,
                "embedding_duration_ms": embed_duration,
                "total_duration_ms": (time.time() - start_time) * 1000,
            }

    # ------------------------------------------------------------------
    # Rich extraction (Docling + images)
    # ------------------------------------------------------------------

    async def _process_rich(
        self, file_path: str, original_filename: str, file_ext: str,
    ) -> List[RichChunk]:
        """Extract text chunks + table chunks + image chunks.

        External Docling is intentionally skipped here: it returns plain text
        with no provenance data, so it cannot produce bounding boxes, separate
        table chunks, or image chunks.  Rich mode always runs in-process Docling.
        """
        try:
            loop = asyncio.get_event_loop()
            rich_chunks = await loop.run_in_executor(
                None,
                self._run_rich_extraction,
                file_path, original_filename,
            )
            # For PDFs, append VLM-captioned image chunks
            if file_ext == ".pdf":
                image_chunks = await self._extract_pdf_images(
                    file_path, original_filename,
                )
                rich_chunks.extend(image_chunks)

            # Recompute chunk_index / total_chunks across the full combined list
            total = len(rich_chunks)
            for i, c in enumerate(rich_chunks):
                c.chunk_index = i
                c.total_chunks = total

            return rich_chunks
        except Exception as e:
            logger.error("Rich extraction failed: %s", e)
            if file_ext == ".pdf":
                content = fallback_pdf_extraction(file_path)
                if content:
                    plain = create_chunks_from_content(
                        content, original_filename, file_ext, self.job_id,
                        text_splitter=self.text_splitter,
                    )
                    return [self._plain_to_rich(c) for c in plain]
            raise

    def _run_rich_extraction(
        self, file_path: str, original_filename: str,
    ) -> List[RichChunk]:
        """Synchronous: run Docling, extract text + table chunks with bboxes."""
        file_ext = os.path.splitext(original_filename)[1].lower()
        fmt_map = get_file_format_map()
        input_format = fmt_map.get(file_ext)
        file_type = file_ext.lstrip(".")

        doc_meta = {"source_type": "file", "file_type": file_type}
        if file_ext == ".pdf":
            doc_meta.update(get_pdf_metadata(file_path))

        # Run Docling converter
        if input_format == InputFormat.PDF:
            result = self._convert_pdf_with_vlm_fallback(file_path)
        else:
            result = self.converter.convert(file_path)

        doc = result.document

        # Collect page dimensions for bbox normalisation
        page_dims: dict[int, tuple[float, float]] = {}
        if hasattr(doc, "pages") and doc.pages:
            for page_no, page in doc.pages.items():
                if hasattr(page, "size") and page.size:
                    page_dims[page_no] = (
                        float(page.size.width),
                        float(page.size.height),
                    )

        # --- Text chunks ---
        text_chunks = self._extract_text_chunks(
            doc, original_filename, file_type, doc_meta, page_dims,
        )

        # --- Table chunks (separate STRUCTURED points) ---
        table_chunks = self._extract_table_chunks(
            doc, original_filename, file_type, doc_meta, page_dims,
            start_index=len(text_chunks),
        )

        all_chunks = text_chunks + table_chunks
        # Fix total_chunks after combining
        total = len(all_chunks)
        for i, c in enumerate(all_chunks):
            c.chunk_index = i
            c.total_chunks = total

        logger.info(
            "Rich extraction: %d text + %d table chunks from %s",
            len(text_chunks), len(table_chunks), original_filename,
        )
        return all_chunks

    def _extract_text_chunks(
        self,
        doc,
        original_filename: str,
        file_type: str,
        doc_meta: dict,
        page_dims: dict,
    ) -> List[RichChunk]:
        chunker = self.chunker
        if chunker is None:
            try:
                from docling.chunking import HierarchicalChunker
                chunker = HierarchicalChunker()
            except ImportError:
                chunker = None

        if chunker is not None:
            raw_chunks = list(chunker.chunk(dl_doc=doc))
            total = len(raw_chunks)
            rich = []
            for i, chunk_data in enumerate(raw_chunks):
                text = chunker.contextualize(chunk=chunk_data)
                if not text or not text.strip():
                    continue

                page_no, bbox, pw, ph = self._prov_info(chunk_data, page_dims)
                text_type = self._infer_text_type(chunk_data)

                rich.append(RichChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=text,
                    source_file=original_filename,
                    chunk_index=i,
                    total_chunks=total,
                    page_number=page_no,
                    page_width=pw,
                    page_height=ph,
                    content_type=CONTENT_TYPE_TEXT,
                    text_type=text_type,
                    text_location=bbox,
                    metadata={**doc_meta},
                ))
            return rich
        else:
            # Fallback: paragraph split
            md = getattr(doc, "export_to_markdown", lambda: "")()
            parts = [p.strip() for p in (md or "").split("\n\n") if p.strip()]
            return [
                RichChunk(
                    id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                    content=text,
                    source_file=original_filename,
                    chunk_index=i,
                    total_chunks=len(parts),
                    content_type=CONTENT_TYPE_TEXT,
                    text_type="body",
                    metadata={**doc_meta},
                )
                for i, text in enumerate(parts)
            ]

    def _extract_table_chunks(
        self,
        doc,
        original_filename: str,
        file_type: str,
        doc_meta: dict,
        page_dims: dict,
        start_index: int,
    ) -> List[RichChunk]:
        tables = getattr(doc, "tables", None) or []
        chunks = []
        for table in tables:
            try:
                table_md = table.export_to_markdown()
            except Exception:
                try:
                    table_md = str(table)
                except Exception:
                    continue
            if not table_md or not table_md.strip():
                continue

            page_no, bbox, pw, ph = self._prov_info(table, page_dims)
            caption = ""
            if hasattr(table, "captions") and table.captions:
                try:
                    caption = table.captions[0].text
                except Exception:
                    pass

            embed_text = f"{caption}\n\n{table_md}".strip() if caption else table_md

            chunks.append(RichChunk(
                id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                content=embed_text,
                source_file=original_filename,
                chunk_index=start_index + len(chunks),
                total_chunks=0,  # fixed later
                page_number=page_no,
                page_width=pw,
                page_height=ph,
                content_type=CONTENT_TYPE_STRUCTURED,
                text_type="table",
                text_location=bbox,
                table_content=table_md,
                table_format="markdown",
                metadata={**doc_meta},
            ))
        return chunks

    async def _extract_pdf_images(
        self, file_path: str, original_filename: str,
    ) -> List[RichChunk]:
        """Extract images from PDF via pypdfium2, caption with VLM."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            logger.warning("pypdfium2 not available; skipping image extraction")
            return []

        file_type = os.path.splitext(original_filename)[1].lstrip(".")
        doc_meta = {"source_type": "file", "file_type": file_type}
        chunks: List[RichChunk] = []

        try:
            pdf = pdfium.PdfDocument(file_path)
        except Exception as e:
            logger.warning("pypdfium2 failed to open %s: %s", original_filename, e)
            return []

        # Collect (image, bbox, page_no, page_w, page_h) tuples first, then caption in parallel
        candidates = []
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            page_width = page.get_width()
            page_height = page.get_height()

            for obj in page.get_objects():
                if obj.type != pdfium.PdfObjectType.IMAGE:
                    continue
                try:
                    bitmap = obj.get_bitmap(render=True)
                    pil_image = bitmap.to_pil()
                except Exception as e:
                    logger.debug("Image extraction failed page %d: %s", page_idx + 1, e)
                    continue

                # Skip tiny images (icons, decorations)
                if pil_image.width < 50 or pil_image.height < 50:
                    continue

                try:
                    bbox_obj = obj.get_pos()
                    bbox = [
                        float(bbox_obj.left),
                        float(bbox_obj.top),
                        float(bbox_obj.right),
                        float(bbox_obj.bottom),
                    ]
                except Exception:
                    bbox = None

                candidates.append((pil_image, bbox, page_idx + 1, page_width, page_height))

        pdf.close()

        if not candidates:
            return []

        # Caption all images concurrently with bounded parallelism (max 4 in-flight)
        semaphore = asyncio.Semaphore(4)

        async def caption_with_sem(pil_image, bbox, page_no, pw, ph):
            async with semaphore:
                caption = await self._caption_image(pil_image, original_filename)
            if not caption:
                return None
            return RichChunk(
                id=f"{self.job_id}_{uuid.uuid4().hex[:8]}",
                content=caption,
                source_file=original_filename,
                chunk_index=0,    # recomputed by caller
                total_chunks=0,
                page_number=page_no,
                page_width=pw,
                page_height=ph,
                content_type=CONTENT_TYPE_IMAGE,
                text_type="picture",
                text_location=bbox,
                image_caption=caption,
                metadata={**doc_meta},
            )

        results = await asyncio.gather(
            *[caption_with_sem(img, bbox, pno, pw, ph) for img, bbox, pno, pw, ph in candidates]
        )
        chunks = [r for r in results if r is not None]
        logger.info("Extracted %d image chunks from %s", len(chunks), original_filename)
        return chunks

    def _resize_for_vlm(self, pil_image):
        """Resize image to fit VLM_INPUT_MAX_SIDE (matches video pipeline behaviour)."""
        max_side = getattr(settings, "VLM_INPUT_MAX_SIDE", 0) or 640
        w, h = pil_image.width, pil_image.height
        if max(w, h) <= max_side:
            return pil_image
        scale = max_side / max(w, h)
        return pil_image.resize((int(w * scale), int(h * scale)))

    async def _caption_image(self, pil_image, filename: str) -> str:
        """Resize image, encode as JPEG, send to VLM, return caption text."""
        try:
            pil_image = self._resize_for_vlm(pil_image)
            buf = io.BytesIO()
            # JPEG is significantly smaller than PNG for photographs/diagrams
            pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image concisely for a search index. "
                                    "Include: what is shown, any text visible, charts or tables present, "
                                    "and any key data points. Be factual and specific."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("VLM captioning failed (%s): %s", filename, e)
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_pdf_with_vlm_fallback(self, file_path: str):
        """Convert PDF using standard Docling (table-accurate). VLM pipeline is used
        for video ingestion, not here — images are handled separately via pypdfium2."""
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
            return converter.convert(file_path)
        except Exception as e:
            logger.warning("PDF conversion with table struct failed (%s), fallback", e)
            return self.converter.convert(file_path)

    def _prov_info(
        self, item, page_dims: dict
    ) -> tuple[Optional[int], Optional[list], Optional[float], Optional[float]]:
        """Extract page_no, bbox [l,t,r,b], page_width, page_height from a Docling item."""
        prov = getattr(item, "prov", None)
        if not prov or not len(prov):
            return None, None, None, None

        p = prov[0]
        page_no = getattr(p, "page_no", None)
        bbox_obj = getattr(p, "bbox", None)
        bbox = None
        if bbox_obj is not None:
            try:
                bbox = [
                    float(getattr(bbox_obj, "l", 0)),
                    float(getattr(bbox_obj, "t", 0)),
                    float(getattr(bbox_obj, "r", 0)),
                    float(getattr(bbox_obj, "b", 0)),
                ]
            except Exception:
                bbox = None

        pw, ph = None, None
        if page_no is not None and page_no in page_dims:
            pw, ph = page_dims[page_no]

        return page_no, bbox, pw, ph

    def _infer_text_type(self, chunk_data) -> str:
        """Infer text_type from Docling chunk label / segment type."""
        label = ""
        # Try chunk label
        if hasattr(chunk_data, "label"):
            label = str(chunk_data.label).lower()
        # Try meta
        elif hasattr(chunk_data, "meta") and hasattr(chunk_data.meta, "doc_items"):
            for item in (chunk_data.meta.doc_items or []):
                lbl = getattr(item, "label", "")
                if lbl:
                    label = str(lbl).lower()
                    break

        if any(h in label for h in ("heading", "title", "section_header")):
            return "header"
        if "caption" in label:
            return "caption"
        if "footnote" in label:
            return "footnote"
        if "list" in label:
            return "list_item"
        if "table" in label:
            return "table"
        if "picture" in label or "figure" in label:
            return "picture"
        return "body"

    def _plain_to_rich(self, chunk: DocumentChunk) -> RichChunk:
        """Wrap a DocumentChunk (no bbox) into a RichChunk."""
        return RichChunk(
            id=chunk.id,
            content=chunk.content,
            source_file=chunk.source_file,
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            page_number=chunk.metadata.get("page_number"),
            content_type=CONTENT_TYPE_TEXT,
            text_type="body",
            metadata={**chunk.metadata},
        )

    @staticmethod
    def _type_summary(chunks: List[RichChunk]) -> str:
        from collections import Counter
        counts = Counter(c.content_type for c in chunks)
        return ", ".join(f"{v} {k}" for k, v in counts.items())

    # ------------------------------------------------------------------
    # Collection + embed + store
    # ------------------------------------------------------------------

    async def _ensure_collection(self):
        collections = self.qdrant.get_collections()
        if any(c.name == self.collection_name for c in collections.collections):
            return
        try:
            test = self.embedding_client.embeddings.create(
                input=["test"], model=self.embedding_model_id,
            )
            vector_size = len(test.data[0].embedding) if test.data else settings.EMBEDDING_DIMENSION
        except Exception as e:
            vector_size = settings.EMBEDDING_DIMENSION
            logger.warning("Embedding dim detection failed: %s", e)
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    async def _embed_and_store(self, chunks: List[RichChunk]):
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [truncate_for_embedding(c.content or "") for c in batch]
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model_id, input=texts,
            )
            embeddings = [e.embedding for e in response.data]
            points = [
                PointStruct(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id),
                    vector=embedding,
                    payload=self._build_payload(chunk),
                )
                for chunk, embedding in zip(batch, embeddings)
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                "Rich processor: stored %d chunks in %s", len(points), self.collection_name,
            )

    def _build_payload(self, chunk: RichChunk) -> dict:
        """Build Qdrant payload with NeMo-style enriched metadata."""
        payload: dict = {
            # Core fields — backward compatible with existing RAG retrieval
            "content": chunk.content,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "chunk_id": chunk.id,
            "source_type": "file",
            "file_type": chunk.metadata.get("file_type", ""),
            "page_number": chunk.page_number,
            "page": chunk.page_number,  # alias for backward-compat with rag_service / legacy chunks
            # NeMo-style enriched fields
            "content_type": chunk.content_type,       # text | structured | image
            "text_type": chunk.text_type or "body",   # header | body | caption | …
            "text_location": chunk.text_location,      # [l, t, r, b] page points or None
            "page_width": chunk.page_width,            # for bbox normalisation in UI
            "page_height": chunk.page_height,
        }

        # Structured (table) extras
        if chunk.content_type == CONTENT_TYPE_STRUCTURED:
            payload["table_content"] = chunk.table_content
            payload["table_format"] = chunk.table_format or "markdown"

        # Image extras
        if chunk.content_type == CONTENT_TYPE_IMAGE:
            payload["image_caption"] = chunk.image_caption

        # Pass through any extra metadata (pdf author, title, dates, …)
        for k, v in (chunk.metadata or {}).items():
            if k not in payload:
                payload[k] = v

        return payload
