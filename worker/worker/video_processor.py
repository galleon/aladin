"""
Video processor for ingestion worker - wraps video pipeline and handles embedding/storage.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings, truncate_for_embedding
from shared.schemas import IngestionJobStatus
from shared.telemetry import get_tracer, IngestionMetrics
from shared.database import get_db_session, get_backend_models

from .video.video_pipeline import run_video_pipeline
from .video.vlm_backend import DummyVLMBackend, OpenAICompatibleVLMBackend, unpack_vlm_for_log
from .video.schemas import VideoChunk

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes videos for ingestion."""

    def __init__(
        self,
        job_ctx,
        job_id: str,
        collection_name: str,
        embedding_model: str,
        processing_config: dict = None,
    ):
        self.job_ctx = job_ctx
        self.job_id = job_id
        self.collection_name = collection_name
        self.embedding_model = embedding_model
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

    def _get_vlm_backend(
        self,
        vlm_api_base: str | None = None,
        vlm_api_key: str | None = None,
        vlm_model_id: str | None = None,
    ):
        """Get VLM backend from config."""
        api_base = vlm_api_base or settings.VLM_API_BASE
        api_key = vlm_api_key or settings.VLM_API_KEY
        model_id = vlm_model_id or settings.VLM_MODEL

        if api_base:
            return OpenAICompatibleVLMBackend(
                api_base=api_base,
                api_key=api_key,
                model_id=model_id or "gpt-4o",
            )
        else:
            logger.warning("No VLM API configured, using DummyVLMBackend")
            return DummyVLMBackend()

    async def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        collections = self.qdrant.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            # Dynamically detect embedding dimension
            try:
                test_embedding = self.embedding_client.embeddings.create(
                    input=["test"],
                    model=self.embedding_model,
                )
                if test_embedding.data and len(test_embedding.data) > 0:
                    vector_size = len(test_embedding.data[0].embedding)
                    logger.info(
                        f"Detected embedding dimension: {vector_size} for model: {self.embedding_model}"
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

    async def process(
        self,
        file_path: str,
        original_filename: str,
        vlm_api_base: str | None = None,
        vlm_api_key: str | None = None,
        vlm_model_id: str | None = None,
        video_mode: str = "procedure",
        vlm_prompt: str | None = None,
        object_tracker: str = "none",
        enable_ocr: bool = False,
        segment_sec: float = 4.0,
        overlap_sec: float = 1.0,
        num_frames: int = 10,
        deblurrer_name: str | None = None,
    ) -> dict:
        """
        Process a video for ingestion.

        Returns:
            dict with processing results: chunks_created, num_segments, etc.
        """
        with self.tracer.start_as_current_span("video_process") as span:
            span.set_attribute("filename", original_filename)
            span.set_attribute("file_path", file_path)

            start_time = time.time()

            # Ensure collection exists
            await self._ensure_collection()

            # Update status
            await self.job_ctx.update_job_status(
                self.job_id,
                IngestionJobStatus.EXTRACTING,
                progress=10,
                message=f"Processing video {original_filename}...",
            )

            # Initialize VLM backend
            vlm = self._get_vlm_backend(vlm_api_base, vlm_api_key, vlm_model_id)

            # Create temporary output file for JSONL
            # We use delete=False because the video pipeline needs to write to the file
            # after we close the handle, but we ensure cleanup in the finally block
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, dir=tempfile.gettempdir()
            )
            tmp_path = tmp_file.name
            tmp_file.close()  # Close file handle so pipeline can write to it

            try:
                # Run video pipeline
                video_id = f"job_{self.job_id}_{Path(original_filename).stem}"
                summary = run_video_pipeline(
                    path=file_path,
                    out_path=tmp_path,
                    video_id=video_id,
                    mode=video_mode,
                    segment_sec=segment_sec,
                    overlap_sec=overlap_sec,
                    num_frames=num_frames,
                    vlm=vlm,
                    object_tracker=object_tracker or "none",
                    enable_cv=bool(object_tracker and object_tracker != "none"),
                    vlm_prompt=vlm_prompt,
                    enable_ocr=enable_ocr,
                    log_vlm_outcome=settings.LOG_VLM_OUTCOME,
                    log_vlm_outcome_sample_every=settings.LOG_VLM_OUTCOME_SAMPLE_EVERY,
                    log_vlm_review_file=settings.LOG_VLM_OUTCOME_REVIEW_FILE,
                    deblurrer_name=deblurrer_name or settings.DEBLUR_NAME,
                )

                # Read JSONL chunks
                chunks = []
                with open(tmp_path, "r", encoding="utf-8") as f:
                    for chunk_idx, line in enumerate(f):
                        if line.strip():
                            chunk_data = json.loads(line)
                            meta = {
                                "source_file": original_filename,
                                "video_id": chunk_data["video_id"],
                                "uuid": chunk_data["video_id"],
                                "t_start": chunk_data["t_start"],
                                "t_end": chunk_data["t_end"],
                                "start_pts": chunk_data["t_start"],
                                "end_pts": chunk_data["t_end"],
                                "frame_times": chunk_data["frame_times"],
                                "fields": chunk_data["fields"],
                                "hash": chunk_data["hash"],
                            }
                            if chunk_data.get("cv_meta"):
                                meta["cv_meta"] = chunk_data["cv_meta"]
                            chunks.append({"text": chunk_data["index_text"], "metadata": meta})
                            if settings.LOG_VLM_OUTCOME and (
                                chunk_idx % settings.LOG_VLM_OUTCOME_SAMPLE_EVERY == 0
                            ):
                                _fields = chunk_data.get("fields", {})
                                unpacked = unpack_vlm_for_log(_fields)
                                # Truncate for log
                                cap = unpacked.get("caption", "")
                                if len(cap) > 500:
                                    unpacked = {**unpacked, "caption": cap[:500] + "..."}
                                ev = unpacked.get("events", [])
                                if len(ev) > 5:
                                    unpacked = {**unpacked, "events": ev[:5] + [f"... and {len(ev) - 5} more"]}
                                logger.info(
                                    "VLM outcome for chunk (stored in vector store) video_id=%s t_start=%.1f t_end=%.1f chunk_index=%d\n%s",
                                    chunk_data["video_id"],
                                    chunk_data["t_start"],
                                    chunk_data["t_end"],
                                    chunk_idx,
                                    json.dumps(unpacked, indent=2, ensure_ascii=False),
                                )

                logger.info(
                    f"Processed video: {len(chunks)} chunks from {summary.get('num_segments', 0)} segments"
                )

                if not chunks:
                    logger.warning(
                        f"No chunks extracted from video: {original_filename}"
                    )
                    return {"chunks_created": 0, "num_segments": 0}

                # Update status
                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.EMBEDDING,
                    progress=70,
                    message=f"Embedding {len(chunks)} chunks...",
                )

                # Generate embeddings (truncate to fit model context, e.g. 512 tokens)
                embed_start = time.time()
                texts = [truncate_for_embedding(chunk["text"]) for chunk in chunks]

                # Create embeddings (async wrapper)
                loop = asyncio.get_event_loop()
                embeddings_response = await loop.run_in_executor(
                    None,
                    lambda: self.embedding_client.embeddings.create(
                        input=texts,
                        model=self.embedding_model,
                    ),
                )

                embeddings = [item.embedding for item in embeddings_response.data]
                embed_duration = (time.time() - embed_start) * 1000

                # Prepare payloads
                payloads = [
                    {"text": chunk["text"], **chunk["metadata"]} for chunk in chunks
                ]

                # Generate unique IDs
                ids = [str(uuid.uuid4()) for _ in chunks]

                # Upsert to Qdrant
                await self.job_ctx.update_job_status(
                    self.job_id,
                    IngestionJobStatus.STORING,
                    progress=90,
                    message=f"Storing {len(chunks)} chunks in vector store...",
                )

                store_start = time.time()
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(id=id_, vector=vec, payload=payload)
                        for id_, vec, payload in zip(ids, embeddings, payloads)
                    ],
                )
                store_duration = (time.time() - store_start) * 1000

                total_duration = (time.time() - start_time) * 1000

                self.metrics.record_processing_time(embed_duration, "embedding")
                self.metrics.record_chunks_created(len(chunks), self.collection_name)
                self.metrics.record_document_processed(self.collection_name, "video")

                span.set_attribute("chunks_created", len(chunks))
                span.set_attribute("num_segments", summary.get("num_segments", 0))

                return {
                    "chunks_created": len(chunks),
                    "num_segments": summary.get("num_segments", 0),
                    "embedding_duration_ms": embed_duration,
                    "storing_duration_ms": store_duration,
                    "total_duration_ms": total_duration,
                }
            finally:
                # Clean up temp file - use try/except to ensure we always attempt cleanup
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        logger.debug("Cleaned up temp file: %s", tmp_path)
                except OSError as e:
                    logger.warning("Failed to remove temp file %s: %s", tmp_path, e)
