"""
Ingestion API router - enqueues jobs for the ingestion worker.
"""
import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends
from pydantic import BaseModel, Field, HttpUrl

from ..config import settings
from ..services.auth import get_current_user
from ..services.file_validation import get_validation_service, ValidationErrorCode
from ..models import User
from ..arq_client import get_arq_pool, get_redis_client
from ..workers.base import TASK_INGESTION_WEB, TASK_INGESTION_FILE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


# ============================================
# Schemas
# ============================================

class CrawlStrategy(str, Enum):
    BFS = "bfs"
    DFS = "dfs"


class TableStrategy(str, Enum):
    RAW_HTML = "raw_html"
    MARKDOWN = "markdown"
    SUMMARY_AND_HTML = "summary_and_html"


class WebSourceConfig(BaseModel):
    """Configuration for web source crawling."""
    url: HttpUrl = Field(..., description="Starting URL to crawl")
    depth_limit: int = Field(default=2, ge=0, le=10)
    strategy: CrawlStrategy = Field(default=CrawlStrategy.BFS)
    inclusion_patterns: List[str] = Field(default_factory=list)
    exclusion_patterns: List[str] = Field(default_factory=list)
    max_pages: int = Field(default=100, ge=1, le=1000)


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    render_js: bool = Field(default=False)
    wait_for_selector: Optional[str] = Field(default=None)
    wait_timeout: int = Field(default=30, ge=5, le=120)
    extract_tables: bool = Field(default=True)
    table_strategy: TableStrategy = Field(default=TableStrategy.MARKDOWN)
    extract_images: bool = Field(default=False)
    vlm_model: Optional[str] = Field(default=None)
    extract_links: bool = Field(default=True)


class WebIngestionRequest(BaseModel):
    """Request to ingest web content."""
    job_id: Optional[str] = Field(default=None)
    source: WebSourceConfig
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    collection_name: str = Field(..., min_length=1, max_length=100)
    metadata: dict = Field(default_factory=dict)


class IngestionJobStatus(str, Enum):
    QUEUED = "queued"
    CRAWLING = "crawling"
    EXTRACTING = "extracting"
    PARTITIONING = "partitioning"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJobResponse(BaseModel):
    """Response for an ingestion job."""
    job_id: str
    status: IngestionJobStatus
    progress: int = 0
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    pages_processed: int = 0
    pages_total: int = 0
    chunks_created: int = 0
    error_message: Optional[str] = None
    markdown: Optional[str] = None  # Extracted markdown content (when extract_only=True)
    type: Optional[str] = None  # web | file | document | video (from Redis)
    filename: Optional[str] = None  # from Redis (data domain uploads)


class ExtractMarkdownResponse(BaseModel):
    """Response for markdown extraction."""
    filename: str
    file_type: str
    markdown: str
    char_count: int
    metadata: dict = {}


# ============================================
# Endpoints
# ============================================

@router.post("/web", response_model=IngestionJobResponse)
async def ingest_web(
    request: WebIngestionRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Enqueue a web ingestion job.

    Crawls the specified URL and linked pages, extracts content,
    and stores vectors in the specified Qdrant collection.
    """
    job_id = request.job_id or f"web_{uuid.uuid4().hex[:12]}"

    try:
        arq_pool = await get_arq_pool()
        redis_client = await get_redis_client()

        # Store job status
        job_data = {
            "job_id": job_id,
            "status": IngestionJobStatus.QUEUED.value,
            "type": "web",
            "url": str(request.source.url),
            "collection_name": request.collection_name,
            "user_id": str(current_user.id),
            "created_at": datetime.utcnow().isoformat(),
            "progress": "0",
            "pages_processed": "0",
            "pages_total": "0",
            "chunks_created": "0",
        }
        await redis_client.hset(f"job:{job_id}", mapping=job_data)
        await redis_client.expire(f"job:{job_id}", 86400)  # 24h TTL

        # Enqueue job (unified worker dispatches by task_name)
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_WEB,
            job_id=job_id,
            url=str(request.source.url),
            collection_name=request.collection_name,
            source_config=request.source.model_dump(),
            processing_config=request.processing_config.model_dump(),
            metadata={**request.metadata, "user_id": current_user.id},
        )

        logger.info(f"Enqueued web job {job_id} for {request.source.url}")

        return IngestionJobResponse(
            job_id=job_id,
            status=IngestionJobStatus.QUEUED,
            progress=0,
            message=f"Job queued for URL: {request.source.url}",
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Failed to enqueue web job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file", response_model=IngestionJobResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = Form(""),  # Optional when extract_only=True
    job_id: Optional[str] = Form(None),
    extract_tables: bool = Form(True),
    extract_images: bool = Form(False),
    vlm_model: Optional[str] = Form(None),
    extract_only: bool = Form(False),  # If True, return markdown without ingesting
    current_user: User = Depends(get_current_user),
):
    """
    Upload and ingest a file.

    Supported formats: PDF, DOCX, PPTX, MD, TXT, HTML, JSON

    Set `extract_only=True` to get the extracted markdown without ingesting to vector store.
    This is useful for translation or other downstream processing.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".txt", ".html", ".htm", ".json"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Validate collection_name is provided when not extract_only
    if not extract_only and not collection_name:
        raise HTTPException(
            status_code=400,
            detail="collection_name is required when extract_only is False",
        )

    job_id = job_id or f"file_{uuid.uuid4().hex[:12]}"

    try:
        # Read file content first for validation
        content = await file.read()
        
        # Perform comprehensive file validation
        validation_service = get_validation_service()
        validation_result = validation_service.validate_file(
            file_content=content,
            filename=file.filename,
            expected_extension=file_ext,
        )
        
        if not validation_result.is_valid:
            logger.warning(
                "File validation failed",
                filename=file.filename,
                error_code=validation_result.error_code,
                error_message=validation_result.error_message,
            )
            raise HTTPException(
                status_code=400,
                detail=validation_result.error_message,
            )
        
        # Save file
        file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}{file_ext}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(content)

        # If extract_only, extract markdown synchronously and return it
        if extract_only:
            from ..services.document_converter_service import DocumentConverterService

            converter = DocumentConverterService()
            markdown_content = await converter.extract_text_from_file(
                file_path=file_path,
                file_type=file_ext.lstrip("."),
                use_llm=settings.MARKER_USE_LLM,
            )

            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

            logger.info(f"Extracted markdown from {file.filename} ({len(markdown_content)} chars)")

            return IngestionJobResponse(
                job_id=job_id,
                status=IngestionJobStatus.COMPLETED,
                progress=100,
                message=f"Markdown extracted from: {file.filename}",
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                markdown=markdown_content,
            )

        # Otherwise, enqueue for background ingestion
        arq_pool = await get_arq_pool()
        redis_client = await get_redis_client()

        # Store job status
        job_data = {
            "job_id": job_id,
            "status": IngestionJobStatus.QUEUED.value,
            "type": "file",
            "filename": file.filename,
            "file_path": file_path,
            "collection_name": collection_name,
            "user_id": str(current_user.id),
            "created_at": datetime.utcnow().isoformat(),
            "progress": "0",
            "chunks_created": "0",
        }
        await redis_client.hset(f"job:{job_id}", mapping=job_data)
        await redis_client.expire(f"job:{job_id}", 86400)

        # Build processing config
        processing_config = {
            "extract_tables": extract_tables,
            "extract_images": extract_images,
            "vlm_model": vlm_model,
        }

        # Enqueue job (unified worker dispatches by task_name)
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_FILE,
            job_id=job_id,
            file_path=file_path,
            original_filename=file.filename,
            collection_name=collection_name,
            processing_config=processing_config,
        )

        logger.info(f"Enqueued file job {job_id} for {file.filename}")

        return IngestionJobResponse(
            job_id=job_id,
            status=IngestionJobStatus.QUEUED,
            progress=0,
            message=f"Job queued for file: {file.filename}",
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Failed to process file: {e}")
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract", response_model=ExtractMarkdownResponse)
async def extract_markdown(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Extract markdown from a file without ingesting to vector store.

    This is a convenience endpoint for the translation module and other
    downstream processing that needs the raw markdown content.

    Supported formats: PDF, DOCX, PPTX, MD, TXT, HTML, JSON
    """
    from ..services.document_converter_service import DocumentConverterService

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".txt", ".html", ".htm", ".json"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    temp_id = uuid.uuid4().hex[:12]
    file_path = os.path.join(settings.UPLOAD_DIR, f"extract_{temp_id}{file_ext}")

    try:
        # Read and validate file content
        content = await file.read()
        
        # Perform comprehensive file validation
        validation_service = get_validation_service()
        validation_result = validation_service.validate_file(
            file_content=content,
            filename=file.filename,
            expected_extension=file_ext,
        )
        
        if not validation_result.is_valid:
            logger.warning(
                "File validation failed",
                filename=file.filename,
                error_code=validation_result.error_code,
                error_message=validation_result.error_message,
            )
            raise HTTPException(
                status_code=400,
                detail=validation_result.error_message,
            )
        
        # Save file temporarily
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)

        # Extract markdown
        converter = DocumentConverterService()
        markdown_content = await converter.extract_text_from_file(
            file_path=file_path,
            file_type=file_ext.lstrip("."),
            use_llm=settings.MARKER_USE_LLM,
        )

        logger.info(f"Extracted markdown from {file.filename} ({len(markdown_content)} chars)")

        return ExtractMarkdownResponse(
            filename=file.filename or "unknown",
            file_type=file_ext.lstrip("."),
            markdown=markdown_content,
            char_count=len(markdown_content),
            metadata={
                "original_size": len(content),
                "extracted_by": "document_converter_service",
            },
        )

    except Exception as e:
        logger.error(f"Failed to extract markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get the status of an ingestion job."""
    redis_client = await get_redis_client()
    job_data = await redis_client.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return IngestionJobResponse(
        job_id=job_data.get("job_id", job_id),
        status=IngestionJobStatus(job_data.get("status", "queued")),
        progress=int(job_data.get("progress", 0)),
        message=job_data.get("message"),
        created_at=datetime.fromisoformat(job_data.get("created_at", datetime.utcnow().isoformat())),
        started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        pages_processed=int(job_data.get("pages_processed", 0)),
        pages_total=int(job_data.get("pages_total", 0)),
        chunks_created=int(job_data.get("chunks_created", 0)),
        error_message=job_data.get("error_message"),
    )


@router.get("/jobs", response_model=List[IngestionJobResponse])
async def list_jobs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    """List ingestion jobs for the current user."""
    redis_client = await get_redis_client()

    jobs = []
    async for key in redis_client.scan_iter(match="job:*", count=100):
        if len(jobs) >= limit:
            break

        job_data = await redis_client.hgetall(key)
        if not job_data:
            continue

        # Filter by user
        if job_data.get("user_id") != str(current_user.id):
            continue

        # Filter by status
        if status and job_data.get("status") != status:
            continue

        jobs.append(IngestionJobResponse(
            job_id=job_data.get("job_id", ""),
            status=IngestionJobStatus(job_data.get("status", "queued")),
            progress=int(job_data.get("progress", 0)),
            message=job_data.get("message"),
            created_at=datetime.fromisoformat(job_data.get("created_at", datetime.utcnow().isoformat())),
            started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
            completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
            pages_processed=int(job_data.get("pages_processed", 0)),
            pages_total=int(job_data.get("pages_total", 0)),
            chunks_created=int(job_data.get("chunks_created", 0)),
            error_message=job_data.get("error_message"),
            type=job_data.get("type"),
            filename=job_data.get("filename"),
        ))

    # Sort by created_at descending
    jobs.sort(key=lambda x: x.created_at, reverse=True)

    return jobs[:limit]


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel a pending ingestion job."""
    redis_client = await get_redis_client()
    job_data = await redis_client.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check ownership
    if job_data.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized to cancel this job")

    current_status = job_data.get("status")
    if current_status in [IngestionJobStatus.COMPLETED.value, IngestionJobStatus.FAILED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {current_status}",
        )

    await redis_client.hset(f"job:{job_id}", "status", "cancelled")
    await redis_client.hset(f"job:{job_id}", "message", "Job cancelled by user")

    return {"message": f"Job {job_id} cancelled"}


@router.get("/collections")
async def list_collections(
    current_user: User = Depends(get_current_user),
):
    """List available Qdrant collections."""
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        collections = client.get_collections()
        return {
            "collections": [
                {"name": c.name}
                for c in collections.collections
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

