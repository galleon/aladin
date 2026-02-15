"""Data Domain API endpoints."""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy import or_, func, and_
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, DataDomain, Document, Job
from ..schemas import (
    DataDomainCreate,
    DataDomainUpdate,
    DataDomainResponse,
    DataDomainListResponse,
    VideoIngestionDefaults,
    DocumentResponse,
    ChunksListResponse,
    VectorChunkResponse,
)
from ..services.auth import get_current_active_user
from ..services.qdrant_service import qdrant_service
from ..services.document_service import document_service
from ..config import settings, get_video_prompt_library
from ..arq_client import get_arq_pool, get_redis_client
from ..workers.base import TASK_INGESTION_FILE, TASK_INGESTION_VIDEO
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/data-domains", tags=["Data Domains"])


def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


async def _enqueue_document_ingestion(
    db: Session,
    domain: DataDomain,
    document: Document,
    user_id: int,
) -> Job:
    """Enqueue one document for ingestion (video or file). Returns the created Job row."""
    arq_pool = await get_arq_pool()
    redis_client = await get_redis_client()
    file_path = document.filename
    original_filename = document.original_filename
    is_video = document.processing_type == "video" or (
        document.file_type and document.file_type.lower() == "mp4"
    )

    if is_video:
        job_id = f"video_{uuid.uuid4().hex[:12]}"
        payload = {
            "redis_job_id": job_id,
            "file_path": file_path,
            "original_filename": original_filename,
            "collection_name": domain.qdrant_collection,
            "document_id": document.id,
        }
        unified_job = Job(
            job_type="ingestion_video",
            status="pending",
            user_id=user_id,
            document_id=document.id,
            payload=payload,
        )
        db.add(unified_job)
        db.commit()
        db.refresh(unified_job)
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_VIDEO,
            job_id=job_id,
            file_path=file_path,
            original_filename=original_filename,
            collection_name=domain.qdrant_collection,
            embedding_model=domain.embedding_model,
            document_id=document.id,
            vlm_api_base=domain.vlm_api_base,
            vlm_api_key=domain.vlm_api_key,
            vlm_model_id=domain.vlm_model_id,
            video_mode=domain.video_mode or "procedure",
            vlm_prompt=domain.vlm_prompt,
            object_tracker=domain.object_tracker or "none",
            enable_ocr=domain.enable_ocr if domain.enable_ocr is not None else False,
            db_url=settings.DATABASE_URL,
        )
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "type": "video",
            "filename": original_filename,
            "file_path": file_path,
            "collection_name": domain.qdrant_collection,
            "document_id": str(document.id),
            "user_id": str(user_id),
            "created_at": datetime.utcnow().isoformat(),
            "progress": "0",
            "chunks_created": "0",
        }
    else:
        job_id = f"doc_{uuid.uuid4().hex[:12]}"
        payload = {
            "redis_job_id": job_id,
            "file_path": file_path,
            "original_filename": original_filename,
            "collection_name": domain.qdrant_collection,
            "document_id": document.id,
        }
        unified_job = Job(
            job_type="ingestion_file",
            status="pending",
            user_id=user_id,
            document_id=document.id,
            payload=payload,
        )
        db.add(unified_job)
        db.commit()
        db.refresh(unified_job)
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_FILE,
            job_id=job_id,
            file_path=file_path,
            original_filename=original_filename,
            collection_name=domain.qdrant_collection,
            processing_config={"embedding_model_id": domain.embedding_model},
            document_id=document.id,
            db_url=settings.DATABASE_URL,
        )
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "type": "document",
            "filename": original_filename,
            "file_path": file_path,
            "collection_name": domain.qdrant_collection,
            "document_id": str(document.id),
            "user_id": str(user_id),
            "created_at": datetime.utcnow().isoformat(),
            "progress": "0",
            "chunks_created": "0",
        }
    await redis_client.hset(f"job:{job_id}", mapping=job_data)
    await redis_client.expire(f"job:{job_id}", 86400)
    logger.info("Enqueued ingestion job", job_id=job_id, document_id=document.id)
    return unified_job


@router.get("/", response_model=list[DataDomainListResponse])
async def list_data_domains(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)
):
    """List all data domains for the current user."""
    domains = db.query(DataDomain).filter(DataDomain.owner_id == current_user.id).all()

    result = []
    for domain in domains:
        doc_count = (
            db.query(Document).filter(Document.data_domain_id == domain.id).count()
        )
        result.append(
            DataDomainListResponse(
                id=domain.id,
                name=domain.name,
                description=domain.description,
                embedding_model=domain.embedding_model,
                qdrant_collection=domain.qdrant_collection,
                owner_id=domain.owner_id,
                document_count=doc_count,
                created_at=domain.created_at,
            )
        )

    return result


# Default VLM prompt templates (match worker/worker/video/prompts.py); placeholders: {t_start}, {t_end}, {num_frames}, {tracks_json}
DEFAULT_VLM_PROMPT_PROCEDURE = (
    "Analyze this procedure video segment [{t_start}s–{t_end}s] using {num_frames} keyframes. "
    "For each step, provide:\n"
    "- **action**: what is being done\n"
    "- **tools**: tools/equipment used\n"
    "- **result**: outcome or visible state change\n"
    "- **warnings**: safety or quality cues if any\n"
    "- **visible_cues**: text, labels, or indicators on screen\n"
    'Respond with a structured JSON: { "caption": "...", "events": [...], "entities": [...], "notes": [...] }.'
)
DEFAULT_VLM_PROMPT_RACE = (
    "Analyze this race/traffic video segment [{t_start}s–{t_end}s] using {num_frames} keyframes. Provide:\n"
    "- **caption**: scene-level summary\n"
    "- **events**: interactions (e.g. closing gap, overtake attempt, lane change), "
    "with per-track references when possible (e.g. 'Car T07', 'Pedestrian P01')\n"
    "- **entities**: vehicles, pedestrians, or other moving objects with stable IDs if known\n"
    "- **notes**: uncertainty when objects are occluded or leave frame\n"
    'Respond with a structured JSON: { "caption": "...", "events": [...], "entities": [...], "notes": [...] }.'
)


@router.get("/video-defaults", response_model=VideoIngestionDefaults)
async def get_video_ingestion_defaults(
    current_user: User = Depends(get_current_active_user),
):
    """Return default VLM API settings and prompt library (keyword -> template) for the Data Domain create form."""
    prompt_library = get_video_prompt_library()
    return VideoIngestionDefaults(
        vlm_api_base=settings.VLM_API_BASE or "",
        vlm_api_key=settings.VLM_API_KEY or "",
        vlm_model=settings.VLM_MODEL or "",
        default_prompt_procedure=prompt_library.get("procedure", DEFAULT_VLM_PROMPT_PROCEDURE),
        default_prompt_race=prompt_library.get("race", DEFAULT_VLM_PROMPT_RACE),
        prompt_library=prompt_library,
        video_ingestion_available=bool(settings.VLM_API_BASE),
    )


@router.post(
    "/", response_model=DataDomainResponse, status_code=status.HTTP_201_CREATED
)
async def create_data_domain(
    domain_data: DataDomainCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new data domain."""
    # Generate unique collection name
    collection_name = f"domain_{uuid.uuid4().hex[:12]}"

    # Create Qdrant collection
    try:
        qdrant_service.create_collection(collection_name, domain_data.embedding_model)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create vector store collection: {str(e)}",
        )

    # Create database record
    domain = DataDomain(
        name=domain_data.name,
        description=domain_data.description,
        embedding_model=domain_data.embedding_model,
        qdrant_collection=collection_name,
        owner_id=current_user.id,
        vlm_api_base=domain_data.vlm_api_base,
        vlm_api_key=domain_data.vlm_api_key,
        vlm_model_id=domain_data.vlm_model_id,
        video_mode=domain_data.video_mode,
        vlm_prompt=domain_data.vlm_prompt,
        object_tracker=domain_data.object_tracker or "none",
        enable_ocr=domain_data.enable_ocr if domain_data.enable_ocr is not None else False,
    )
    db.add(domain)
    db.commit()
    db.refresh(domain)

    logger.info("Created data domain", domain_id=domain.id, name=domain.name)
    return domain


@router.get("/{domain_id}", response_model=DataDomainResponse)
async def get_data_domain(
    domain_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific data domain."""
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )

    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )

    return domain


CHUNKS_SCROLL_BATCH = 500
CHUNKS_SCROLL_CAP = 10_000


def _sort_key(it: dict) -> tuple:
    """Order by source_file, then chunk_index (documents) or t_start (video)."""
    payload = it.get("payload") or {}
    source = (payload.get("source_file") or "").lower()
    idx = payload.get("chunk_index")
    if idx is None:
        idx = payload.get("t_start", 0)
    return (source, idx if isinstance(idx, (int, float)) else 0)


@router.get("/{domain_id}/chunks", response_model=ChunksListResponse)
async def list_domain_chunks(
    domain_id: int,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List vector store chunks in document order (source_file, chunk_index/t_start) with pagination."""
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )
    try:
        all_items: list = []
        scroll_offset = None
        while len(all_items) < CHUNKS_SCROLL_CAP:
            batch, scroll_offset = qdrant_service.scroll(
                collection_name=domain.qdrant_collection,
                limit=CHUNKS_SCROLL_BATCH,
                offset=scroll_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not batch:
                break
            all_items.extend(batch)
            if scroll_offset is None:
                break
        all_items = all_items[:CHUNKS_SCROLL_CAP]
        all_items.sort(key=_sort_key)
        total = len(all_items)
        page = all_items[offset : offset + limit]
        return ChunksListResponse(
            items=[VectorChunkResponse(id=it["id"], payload=it["payload"]) for it in page],
            total=total,
            has_more=(offset + len(page) < total),
            next_offset=None,
        )
    except Exception as e:
        logger.warning("Scroll failed for domain chunks", domain_id=domain_id, error=str(e))
        return ChunksListResponse(items=[], total=0, has_more=False, next_offset=None)


@router.put("/{domain_id}", response_model=DataDomainResponse)
async def update_data_domain(
    domain_id: int,
    domain_data: DataDomainUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update a data domain."""
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )

    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )

    # Update fields
    if domain_data.name is not None:
        domain.name = domain_data.name
    if domain_data.description is not None:
        domain.description = domain_data.description
    if domain_data.vlm_api_base is not None:
        domain.vlm_api_base = domain_data.vlm_api_base
    if domain_data.vlm_api_key is not None:
        domain.vlm_api_key = domain_data.vlm_api_key
    if domain_data.vlm_model_id is not None:
        domain.vlm_model_id = domain_data.vlm_model_id
    if domain_data.video_mode is not None:
        domain.video_mode = domain_data.video_mode
    vlm_prompt_changed = False
    if domain_data.vlm_prompt is not None:
        if domain.vlm_prompt != domain_data.vlm_prompt:
            vlm_prompt_changed = True
        domain.vlm_prompt = domain_data.vlm_prompt
    if domain_data.object_tracker is not None:
        domain.object_tracker = domain_data.object_tracker
    if domain_data.enable_ocr is not None:
        domain.enable_ocr = domain_data.enable_ocr

    db.commit()
    db.refresh(domain)

    # When VLM prompt changed, destroy embeddings and re-ingest video documents only
    if vlm_prompt_changed:
        video_docs = (
            db.query(Document)
            .filter(
                Document.data_domain_id == domain_id,
                or_(
                    Document.processing_type == "video",
                    and_(Document.file_type.isnot(None), func.lower(Document.file_type) == "mp4"),
                ),
            )
            .all()
        )
        for doc in video_docs:
            try:
                document_service.delete_vectors_by_source_file(
                    domain.qdrant_collection, doc.original_filename
                )
                doc.status = "pending"
                doc.chunk_count = 0
                doc.error_message = None
                db.add(doc)
                db.commit()
                db.refresh(doc)
                await _enqueue_document_ingestion(db, domain, doc, current_user.id)
                logger.info(
                    "Re-ingestion enqueued after prompt change",
                    domain_id=domain_id,
                    document_id=doc.id,
                    original_filename=doc.original_filename,
                )
            except Exception as e:
                logger.warning(
                    "Failed to re-enqueue video document after prompt change",
                    domain_id=domain_id,
                    document_id=doc.id,
                    error=str(e),
                )
                doc.status = "failed"
                doc.error_message = str(e)
                db.add(doc)
                db.commit()

    return domain


@router.delete("/{domain_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_domain(
    domain_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a data domain: all chunks (Qdrant collection), document files, and DB records."""
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )

    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )

    # Delete the domain's Qdrant collection (and all chunks in it)
    if not qdrant_service.delete_collection(domain.qdrant_collection):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to delete vector collection; data domain not deleted",
        )

    # Delete files from disk
    for doc in domain.documents:
        document_service.delete_file(doc.filename)

    # Delete from database (cascades to documents; agent_data_domains rows removed via FK CASCADE)
    db.delete(domain)
    db.commit()

    logger.info("Deleted data domain", domain_id=domain_id)


@router.post("/{domain_id}/documents", response_model=DocumentResponse)
async def upload_document(
    domain_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Upload a document to a data domain."""
    # Check domain exists and belongs to user
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )

    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )

    # Validate file extension
    file_ext = get_file_extension(file.filename)
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read file content
    content = await file.read()

    # Check file size (videos have larger limit)
    is_video = file_ext == "mp4"
    max_size = settings.MAX_VIDEO_SIZE if is_video else settings.MAX_FILE_SIZE
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {max_size / 1024 / 1024}MB",
        )

    # Save file
    file_path = document_service.save_uploaded_file(content, file.filename)

    # Create document record
    document = Document(
        filename=file_path,
        original_filename=file.filename,
        file_type=file_ext,
        file_size=len(content),
        status="pending",
        processing_type="video" if is_video else "document",
        data_domain_id=domain_id,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Start background processing
    if is_video:
        # Video processing - enqueue to ingestion worker
        try:
            job_id = f"video_{uuid.uuid4().hex[:12]}"
            arq_pool = await get_arq_pool()
            redis_client = await get_redis_client()

            # Create unified Job row so it appears in Jobs UI
            payload = {
                "redis_job_id": job_id,
                "file_path": file_path,
                "original_filename": file.filename,
                "collection_name": domain.qdrant_collection,
                "document_id": document.id,
            }
            unified_job = Job(
                job_type="ingestion_video",
                status="pending",
                user_id=current_user.id,
                document_id=document.id,
                payload=payload,
            )
            db.add(unified_job)
            db.commit()
            db.refresh(unified_job)

            # Enqueue first so we only show "queued" when the job is actually in the queue
            await arq_pool.enqueue_job(
                "run_task",
                task_name=TASK_INGESTION_VIDEO,
                job_id=job_id,
                file_path=file_path,
                original_filename=file.filename,
                collection_name=domain.qdrant_collection,
                embedding_model=domain.embedding_model,
                document_id=document.id,
                vlm_api_base=domain.vlm_api_base,
                vlm_api_key=domain.vlm_api_key,
                vlm_model_id=domain.vlm_model_id,
                video_mode=domain.video_mode or "procedure",
                vlm_prompt=domain.vlm_prompt,
                object_tracker=domain.object_tracker or "none",
                enable_ocr=domain.enable_ocr if domain.enable_ocr is not None else False,
                db_url=settings.DATABASE_URL,
            )

            # Store job status in Redis (only after enqueue succeeds)
            job_data = {
                "job_id": job_id,
                "status": "queued",
                "type": "video",
                "filename": file.filename,
                "file_path": file_path,
                "collection_name": domain.qdrant_collection,
                "document_id": str(document.id),
                "user_id": str(current_user.id),
                "created_at": datetime.utcnow().isoformat(),
                "progress": "0",
                "chunks_created": "0",
            }
            await redis_client.hset(f"job:{job_id}", mapping=job_data)
            await redis_client.expire(f"job:{job_id}", 86400)  # 24h TTL

            logger.info(
                "Video job enqueued",
                job_id=job_id,
                document_id=document.id,
                filename=file.filename,
            )
        except Exception as e:
            # Update document status to failed
            doc = db.query(Document).filter(Document.id == document.id).first()
            if doc:
                doc.status = "failed"
                doc.error_message = f"Failed to enqueue video job: {str(e)}"
                db.commit()
            # Update Job row and Redis so Jobs UI shows failed, not "queued"
            job_row = db.query(Job).filter(Job.id == unified_job.id).first()
            if job_row:
                job_row.status = "failed"
                job_row.error_message = "Failed to enqueue"
                db.add(job_row)
            try:
                r = await get_redis_client()
                await r.hset(f"job:{job_id}", "status", "failed")
                await r.hset(f"job:{job_id}", "message", "Failed to enqueue")
            except Exception:
                pass
            db.commit()
            logger.error(
                "Failed to enqueue video job", document_id=document.id, error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to enqueue video processing job: {str(e)}",
            )
    else:
        # Document processing - enqueue to ingestion worker (same as video)
        try:
            job_id = f"doc_{uuid.uuid4().hex[:12]}"
            arq_pool = await get_arq_pool()
            redis_client = await get_redis_client()

            # Create unified Job row so it appears in Jobs UI
            payload = {
                "redis_job_id": job_id,
                "file_path": file_path,
                "original_filename": file.filename,
                "collection_name": domain.qdrant_collection,
                "document_id": document.id,
            }
            unified_job = Job(
                job_type="ingestion_file",
                status="pending",
                user_id=current_user.id,
                document_id=document.id,
                payload=payload,
            )
            db.add(unified_job)
            db.commit()
            db.refresh(unified_job)

            # Enqueue first so we only show "queued" when the job is actually in the queue
            await arq_pool.enqueue_job(
                "run_task",
                task_name=TASK_INGESTION_FILE,
                job_id=job_id,
                file_path=file_path,
                original_filename=file.filename,
                collection_name=domain.qdrant_collection,
                processing_config={"embedding_model_id": domain.embedding_model},
                document_id=document.id,
                db_url=settings.DATABASE_URL,
            )

            # Store job status in Redis (only after enqueue succeeds)
            job_data = {
                "job_id": job_id,
                "status": "queued",
                "type": "document",
                "filename": file.filename,
                "file_path": file_path,
                "collection_name": domain.qdrant_collection,
                "document_id": str(document.id),
                "user_id": str(current_user.id),
                "created_at": datetime.utcnow().isoformat(),
                "progress": "0",
                "chunks_created": "0",
            }
            await redis_client.hset(f"job:{job_id}", mapping=job_data)
            await redis_client.expire(f"job:{job_id}", 86400)

            logger.info(
                "Document job enqueued",
                job_id=job_id,
                document_id=document.id,
                filename=file.filename,
            )
        except Exception as e:
            doc = db.query(Document).filter(Document.id == document.id).first()
            if doc:
                doc.status = "failed"
                doc.error_message = f"Failed to enqueue job: {str(e)}"
                db.commit()
            # Update Job row and Redis so Jobs UI shows failed, not "queued"
            job_row = db.query(Job).filter(Job.id == unified_job.id).first()
            if job_row:
                job_row.status = "failed"
                job_row.error_message = "Failed to enqueue"
                db.add(job_row)
            try:
                r = await get_redis_client()
                await r.hset(f"job:{job_id}", "status", "failed")
                await r.hset(f"job:{job_id}", "message", "Failed to enqueue")
            except Exception:
                pass
            db.commit()
            logger.error(
                "Failed to enqueue document job", document_id=document.id, error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to enqueue document processing job: {str(e)}",
            )

    return document


@router.post("/{domain_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_data_domain(
    domain_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Reindex full data domain: clear vector store and re-enqueue ingestion for all documents."""
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )
    # Delete and recreate collection so it is empty
    qdrant_service.delete_collection(domain.qdrant_collection)
    qdrant_service.create_collection(domain.qdrant_collection, domain.embedding_model)
    # Reset all documents and enqueue each for ingestion
    for doc in domain.documents:
        doc.status = "pending"
        doc.chunk_count = 0
        doc.error_message = None
    db.commit()
    for doc in domain.documents:
        try:
            await _enqueue_document_ingestion(db, domain, doc, current_user.id)
        except Exception as e:
            logger.warning("Failed to enqueue document for reindex", document_id=doc.id, error=str(e))
            doc.status = "failed"
            doc.error_message = str(e)
            db.commit()
    logger.info("Reindex domain enqueued", domain_id=domain_id, documents_count=len(domain.documents))
    return {"message": "Reindex started", "documents_queued": len(domain.documents)}


@router.post(
    "/{domain_id}/documents/{document_id}/reindex",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_document(
    domain_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Reindex single document: remove its chunks from the vector store and re-enqueue ingestion."""
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.data_domain_id == domain_id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    # Remove chunks for this document (by source_file; worker stores original_filename as source_file)
    document_service.delete_vectors_by_source_file(
        domain.qdrant_collection, document.original_filename
    )
    document_service.delete_document_vectors(document_id, domain.qdrant_collection)
    # Reset document and enqueue
    document.status = "pending"
    document.chunk_count = 0
    document.error_message = None
    db.commit()
    try:
        await _enqueue_document_ingestion(db, domain, document, current_user.id)
    except Exception as e:
        document.status = "failed"
        document.error_message = str(e)
        db.commit()
        logger.error("Failed to enqueue document reindex", document_id=document_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue reindex: {str(e)}",
        )
    logger.info("Reindex document enqueued", document_id=document_id)
    return {"message": "Reindex started", "document_id": document_id}


@router.delete(
    "/{domain_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(
    domain_id: int,
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a document from a data domain."""
    # Check domain exists and belongs to user
    domain = (
        db.query(DataDomain)
        .filter(DataDomain.id == domain_id, DataDomain.owner_id == current_user.id)
        .first()
    )

    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data domain not found"
        )

    # Get document
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.data_domain_id == domain_id)
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    # Delete chunks from Qdrant (by document_id and by source_file; workers store source_file)
    document_service.delete_document_vectors(document_id, domain.qdrant_collection)
    document_service.delete_vectors_by_source_file(
        domain.qdrant_collection, document.original_filename
    )

    # Delete file from disk
    document_service.delete_file(document.filename)

    # Delete from database
    db.delete(document)
    db.commit()

    logger.info("Deleted document", document_id=document_id)
