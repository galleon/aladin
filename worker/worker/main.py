"""
Ingestion Worker - ARQ worker for processing ingestion jobs.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from arq import cron
from arq.connections import RedisSettings

# Add shared to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.schemas import IngestionJobStatus
from shared.telemetry import setup_telemetry, get_tracer, IngestionMetrics

from .web import WebProcessor
from .file import FileProcessor
from .video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class JobContext:
    """Context for job processing with Redis connection."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.tracer = get_tracer()
        self.metrics = IngestionMetrics()

    async def update_job_status(
        self,
        job_id: str,
        status: IngestionJobStatus,
        progress: int = None,
        message: str = None,
        **kwargs,
    ):
        """Update job status in Redis."""
        updates = {"status": status.value}

        if progress is not None:
            updates["progress"] = str(progress)
        if message:
            updates["message"] = message

        for key, value in kwargs.items():
            if value is not None:
                updates[key] = str(value) if not isinstance(value, str) else value

        await self.redis.hset(f"job:{job_id}", mapping=updates)

    async def mark_job_started(self, job_id: str):
        """Mark job as started."""
        await self.update_job_status(
            job_id,
            IngestionJobStatus.EXTRACTING,
            progress=5,
            started_at=datetime.utcnow().isoformat(),
        )

    async def mark_job_completed(self, job_id: str, chunks_created: int):
        """Mark job as completed."""
        await self.update_job_status(
            job_id,
            IngestionJobStatus.COMPLETED,
            progress=100,
            message="Job completed successfully",
            completed_at=datetime.utcnow().isoformat(),
            chunks_created=chunks_created,
        )

    async def mark_job_failed(self, job_id: str, error: str):
        """Mark job as failed."""
        await self.update_job_status(
            job_id,
            IngestionJobStatus.FAILED,
            message="Job failed",
            error_message=error,
            completed_at=datetime.utcnow().isoformat(),
        )


async def process_web_job(
    ctx: dict,
    job_id: str,
    url: str,
    collection_name: str,
    source_config: dict,
    processing_config: dict,
    metadata: dict = None,
):
    """
    Process a web ingestion job.

    Uses crawl4ai to crawl web pages and extract content.
    """
    job_ctx = JobContext(ctx["redis"])
    tracer = get_tracer()

    with tracer.start_as_current_span("process_web_job") as span:
        span.set_attribute("job_id", job_id)
        span.set_attribute("url", url)
        span.set_attribute("collection", collection_name)

        try:
            await job_ctx.mark_job_started(job_id)
            logger.info(f"Processing web job {job_id} for {url}")

            # Initialize processor
            processor = WebProcessor(
                job_ctx=job_ctx,
                job_id=job_id,
                collection_name=collection_name,
            )

            # Process the web content
            result = await processor.process(
                url=url,
                source_config=source_config,
                processing_config=processing_config,
                metadata=metadata or {},
            )

            await job_ctx.mark_job_completed(job_id, result["chunks_created"])
            logger.info(
                f"Web job {job_id} completed: {result['chunks_created']} chunks created"
            )

            return result

        except Exception as e:
            logger.error(f"Web job {job_id} failed: {e}")
            span.record_exception(e)
            await job_ctx.mark_job_failed(job_id, str(e))
            raise


async def process_file_job(
    ctx: dict,
    job_id: str,
    file_path: str,
    original_filename: str,
    collection_name: str,
    processing_config: dict = None,
    document_id: int | None = None,
    db_url: str | None = None,
):
    """
    Process a file ingestion job.

    Uses docling to extract content from various document formats.
    When document_id (and db_url) are provided, updates the backend Document
    record (data domain upload flow). Otherwise ingestion-API flow; file is
    deleted after processing.
    """
    from shared.database import get_db_session, get_backend_models

    job_ctx = JobContext(ctx["redis"])
    tracer = get_tracer()
    db = None
    if document_id is not None:
        Document, _ = get_backend_models()
        db = get_db_session()

    with tracer.start_as_current_span("process_file_job") as span:
        span.set_attribute("job_id", job_id)
        span.set_attribute("filename", original_filename)
        span.set_attribute("collection", collection_name)
        if document_id is not None:
            span.set_attribute("document_id", document_id)

        try:
            await job_ctx.mark_job_started(job_id)
            logger.info(f"Processing file job {job_id} for {original_filename}")

            # Update document status to processing (data domain flow)
            if db and document_id is not None:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.status = "processing"
                    db.commit()
                    logger.info(f"Updated document {document_id} status to processing")

            # Initialize processor
            processor = FileProcessor(
                job_ctx=job_ctx,
                job_id=job_id,
                collection_name=collection_name,
                processing_config=processing_config or {},
            )

            # Process the file
            result = await processor.process(
                file_path=file_path,
                original_filename=original_filename,
                processing_config=processing_config or {},
            )

            # Update document status to ready (data domain flow)
            if db and document_id is not None:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.status = "ready"
                    doc.chunk_count = result["chunks_created"]
                    db.commit()
                    logger.info(
                        f"Updated document {document_id} status to ready with {result['chunks_created']} chunks"
                    )

            await job_ctx.mark_job_completed(job_id, result["chunks_created"])
            logger.info(
                f"File job {job_id} completed: {result['chunks_created']} chunks created"
            )

            # Clean up uploaded file only for ingestion-API flow (no Document record)
            if document_id is None and os.path.exists(file_path):
                os.remove(file_path)

            return result

        except Exception as e:
            logger.error(f"File job {job_id} failed: {e}")
            span.record_exception(e)
            await job_ctx.mark_job_failed(job_id, str(e))

            # Update document status to failed (data domain flow)
            if db and document_id is not None:
                try:
                    doc = db.query(Document).filter(Document.id == document_id).first()
                    if doc:
                        doc.status = "failed"
                        doc.error_message = str(e)
                        db.commit()
                        logger.info(f"Updated document {document_id} status to failed")
                except Exception as db_error:
                    logger.error(f"Failed to update document status: {db_error}")
            raise
        finally:
            if db:
                db.close()


async def process_video_job(
    ctx: dict,
    job_id: str,
    file_path: str,
    original_filename: str,
    collection_name: str,
    embedding_model: str,
    document_id: int,
    vlm_api_base: str | None = None,
    vlm_api_key: str | None = None,
    vlm_model_id: str | None = None,
    video_mode: str = "procedure",
    vlm_prompt: str | None = None,
    object_tracker: str = "none",
    enable_ocr: bool = False,
    db_url: str | None = None,
):
    """
    Process a video ingestion job.

    Uses video pipeline to segment, analyze with VLM, and store chunks in Qdrant.
    Updates document status in PostgreSQL.
    """
    from shared.database import get_db_session, get_backend_models

    job_ctx = JobContext(ctx["redis"])
    tracer = get_tracer()
    Document, DataDomain = get_backend_models()

    with tracer.start_as_current_span("process_video_job") as span:
        span.set_attribute("job_id", job_id)
        span.set_attribute("filename", original_filename)
        span.set_attribute("collection", collection_name)
        span.set_attribute("document_id", document_id)

        db = None
        try:
            await job_ctx.mark_job_started(job_id)
            logger.info(f"Processing video job {job_id} for {original_filename}")

            # Update document status to processing
            db = get_db_session()
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = "processing"
                db.commit()
                logger.info(f"Updated document {document_id} status to processing")

            # Initialize processor
            processor = VideoProcessor(
                job_ctx=job_ctx,
                job_id=job_id,
                collection_name=collection_name,
                embedding_model=embedding_model,
            )

            # Process the video
            result = await processor.process(
                file_path=file_path,
                original_filename=original_filename,
                vlm_api_base=vlm_api_base,
                vlm_api_key=vlm_api_key,
                vlm_model_id=vlm_model_id,
                video_mode=video_mode,
                vlm_prompt=vlm_prompt,
                object_tracker=object_tracker or "none",
                enable_ocr=enable_ocr,
            )

            # Update document status to ready
            if db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.status = "ready"
                    doc.chunk_count = result["chunks_created"]
                    db.commit()
                    logger.info(
                        f"Updated document {document_id} status to ready with {result['chunks_created']} chunks"
                    )

            await job_ctx.mark_job_completed(job_id, result["chunks_created"])
            logger.info(
                f"Video job {job_id} completed: {result['chunks_created']} chunks created"
            )

            return result

        except Exception as e:
            logger.error(f"Video job {job_id} failed: {e}")
            span.record_exception(e)
            await job_ctx.mark_job_failed(job_id, str(e))

            # Update document status to failed
            if db:
                try:
                    doc = db.query(Document).filter(Document.id == document_id).first()
                    if doc:
                        doc.status = "failed"
                        doc.error_message = str(e)
                        db.commit()
                        logger.info(f"Updated document {document_id} status to failed")
                except Exception as db_error:
                    logger.error(f"Failed to update document status: {db_error}")

            raise
        finally:
            if db:
                db.close()


async def startup(ctx: dict):
    """Worker startup handler."""
    logger.info("Starting Ingestion Worker...")

    # Set up OpenTelemetry
    setup_telemetry("ingestion-worker", "1.0.0")

    # Create Redis client
    ctx["redis"] = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True,
    )

    logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    if getattr(settings, "LOG_VLM_OUTCOME", False):
        logger.info(
            "VLM outcome logging enabled (LOG_VLM_OUTCOME=true sample_every=%s review_file=%s)",
            getattr(settings, "LOG_VLM_OUTCOME_SAMPLE_EVERY", 1),
            getattr(settings, "LOG_VLM_OUTCOME_REVIEW_FILE", False),
        )


async def shutdown(ctx: dict):
    """Worker shutdown handler."""
    logger.info("Shutting down Ingestion Worker...")
    await ctx["redis"].close()


class WorkerSettings:
    """ARQ worker settings."""

    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        database=settings.REDIS_DB,
    )

    functions = [
        process_web_job,
        process_file_job,
        process_video_job,
    ]

    on_startup = startup
    on_shutdown = shutdown

    # Worker configuration
    max_jobs = settings.WORKER_CONCURRENCY
    job_timeout = settings.JOB_TIMEOUT

    # Retry configuration
    max_tries = 3
    retry_delay = 10  # seconds

    # Health check
    health_check_interval = 30


if __name__ == "__main__":
    from arq import run_worker

    run_worker(WorkerSettings)
