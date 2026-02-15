"""
Unified jobs API: list jobs from the jobs table and queue status.

Admin UI uses this for queue status and a single table of jobs (id, job_type, status, date submitted, duration).
Actions: requeue (pending/failed), cancel (pending/processing), delete.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..arq_client import (
    get_arq_pool,
    get_redis_client,
    DEFAULT_QUEUE_NAME,
    KNOWN_QUEUES,
)
from ..config import settings
from ..database import get_db
from ..models import Job, User, Document, DataDomain
from ..services.auth import get_current_active_user
from ..workers.base import TASK_INGESTION_FILE, TASK_INGESTION_VIDEO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

QUEUE_STATUS_TIMEOUT = 2.0


# ============== Schemas ==============


class UnifiedJobListItem(BaseModel):
    """Single row for the unified jobs table (admin UI)."""

    id: int
    job_type: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    class Config:
        from_attributes = True


class QueueStatusResponse(BaseModel):
    """Queue status (Redis)."""

    queue_name: str
    queued_count: Optional[int] = None
    error: Optional[str] = None


class QueueInfo(BaseModel):
    name: str
    label: str
    count: Optional[int] = None


class QueuesResponse(BaseModel):
    queues: List[QueueInfo]
    error: Optional[str] = None


# ============== Endpoints ==============


@router.get("", response_model=List[UnifiedJobListItem])
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job_type"),
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List current user's jobs from the unified jobs table.
    For ingestion jobs, status/completed_at are merged from Redis so the UI shows live progress.
    Returns: job id, job_type, status, created_at, completed_at, duration_seconds.
    """
    q = db.query(Job).filter(Job.user_id == current_user.id)
    if job_type:
        q = q.filter(Job.job_type == job_type)
    if status_filter:
        q = q.filter(Job.status == status_filter)
    rows = q.order_by(Job.created_at.desc()).limit(limit).all()

    redis_client = None
    try:
        redis_client = await get_redis_client()
    except Exception:
        pass

    result = []
    payload = None
    for j in rows:
        status = j.status
        completed_at = j.completed_at
        payload = j.payload if isinstance(j.payload, dict) else {}
        # For ingestion jobs, merge live status from Redis (worker updates Redis only)
        if redis_client and payload and j.job_type.startswith("ingestion_"):
            redis_job_id = payload.get("redis_job_id")
            if redis_job_id:
                try:
                    redis_data = await asyncio.wait_for(
                        redis_client.hgetall(f"job:{redis_job_id}"),
                        timeout=1.0,
                    )
                    if redis_data:
                        status = redis_data.get("status") or status
                        completed_at_str = redis_data.get("completed_at")
                        if completed_at_str:
                            try:
                                parsed = datetime.fromisoformat(
                                    completed_at_str.replace("Z", "+00:00")
                                )
                                if parsed.tzinfo:
                                    parsed = parsed.replace(tzinfo=None)
                                completed_at = parsed
                            except Exception:
                                pass
                except (asyncio.TimeoutError, Exception):
                    pass

        duration_seconds = None
        if completed_at is not None and j.created_at is not None:
            try:
                end = (
                    completed_at.replace(tzinfo=None)
                    if getattr(completed_at, "tzinfo", None)
                    else completed_at
                )
                start = (
                    j.created_at.replace(tzinfo=None)
                    if getattr(j.created_at, "tzinfo", None)
                    else j.created_at
                )
                duration_seconds = (end - start).total_seconds()
            except (TypeError, AttributeError):
                pass

        result.append(
            UnifiedJobListItem(
                id=j.id,
                job_type=j.job_type,
                status=status,
                created_at=j.created_at,
                completed_at=completed_at,
                duration_seconds=duration_seconds,
            )
        )
    return result


@router.get("/queue-status", response_model=QueueStatusResponse)
async def get_queue_status(
    current_user: User = Depends(get_current_active_user),
):
    """Number of jobs waiting in the Redis job queue (arq:queue)."""
    try:
        redis_client = await get_redis_client()
        queued = await asyncio.wait_for(
            redis_client.zcard(DEFAULT_QUEUE_NAME),
            timeout=QUEUE_STATUS_TIMEOUT,
        )
        return QueueStatusResponse(queue_name=DEFAULT_QUEUE_NAME, queued_count=queued)
    except asyncio.TimeoutError:
        logger.warning("Queue status timed out", timeout=QUEUE_STATUS_TIMEOUT)
        return QueueStatusResponse(
            queue_name=DEFAULT_QUEUE_NAME,
            queued_count=None,
            error="timeout",
        )
    except Exception as e:
        logger.warning("Queue status check failed", error=str(e))
        return QueueStatusResponse(
            queue_name=DEFAULT_QUEUE_NAME,
            queued_count=None,
            error=str(e),
        )


@router.get("/queues", response_model=QueuesResponse)
async def list_queues(
    current_user: User = Depends(get_current_active_user),
):
    """List Redis job queues with pending counts."""
    try:
        redis_client = await get_redis_client()
        result = []
        for queue_name, label in KNOWN_QUEUES:
            try:
                count = await asyncio.wait_for(
                    redis_client.zcard(queue_name),
                    timeout=QUEUE_STATUS_TIMEOUT,
                )
            except Exception:
                count = None
            result.append(QueueInfo(name=queue_name, label=label, count=count))
        return QueuesResponse(queues=result)
    except Exception as e:
        logger.warning("List queues failed", error=str(e))
        return QueuesResponse(queues=[], error=str(e))


@router.post("/queues/purge", status_code=status.HTTP_200_OK)
async def purge_queue(
    queue_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """Clear all pending jobs from the Redis job queue (arq:queue)."""
    allowed = {q[0] for q in KNOWN_QUEUES}
    if queue_name not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown queue. Allowed: {list(allowed)}",
        )
    try:
        redis_client = await get_redis_client()
        removed = await redis_client.zcard(queue_name)
        await redis_client.zremrangebyrank(queue_name, 0, -1)
        logger.info(
            "Queue purged",
            queue_name=queue_name,
            removed=removed,
            user_id=current_user.id,
        )
        return {"queue_name": queue_name, "removed": removed}
    except Exception as e:
        logger.warning("Purge queue failed", queue_name=queue_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ============== Job actions ==============


def _get_job_or_404(job_id: int, current_user: User, db: Session) -> Job:
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return job


@router.post("/{job_id}/requeue", status_code=status.HTTP_200_OK)
async def requeue_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Re-queue a job so a worker picks it up again.
    Allowed for pending or failed jobs. Supported: ingestion_file, ingestion_video.
    """
    job = _get_job_or_404(job_id, current_user, db)
    live_status = job.status
    payload = job.payload if isinstance(job.payload, dict) else {}

    if live_status not in ("pending", "failed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only pending or failed jobs can be re-queued; current status is {live_status!r}",
        )

    if job.job_type not in (TASK_INGESTION_FILE, TASK_INGESTION_VIDEO):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Re-queue not supported for job_type={job.job_type!r}",
        )

    redis_job_id = payload.get("redis_job_id")
    document_id = payload.get("document_id") or (
        job.document_id if job.document_id else None
    )
    if not redis_job_id or not document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job payload missing redis_job_id or document_id",
        )

    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    domain = (
        db.query(DataDomain)
        .filter(
            DataDomain.id == document.data_domain_id,
            DataDomain.owner_id == current_user.id,
        )
        .first()
    )
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Data domain not found or access denied",
        )

    file_path = payload.get("file_path") or document.filename
    original_filename = payload.get("original_filename") or document.original_filename
    collection_name = payload.get("collection_name") or domain.qdrant_collection

    redis_client = await get_redis_client()
    job_data = {
        "job_id": redis_job_id,
        "status": "queued",
        "type": "video" if job.job_type == TASK_INGESTION_VIDEO else "document",
        "filename": original_filename,
        "file_path": file_path,
        "collection_name": collection_name,
        "document_id": str(document.id),
        "user_id": str(current_user.id),
        "progress": "0",
        "chunks_created": "0",
    }
    await redis_client.hset(f"job:{redis_job_id}", mapping=job_data)
    await redis_client.expire(f"job:{redis_job_id}", 86400)

    arq_pool = await get_arq_pool()
    if job.job_type == TASK_INGESTION_VIDEO:
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_VIDEO,
            job_id=redis_job_id,
            file_path=file_path,
            original_filename=original_filename,
            collection_name=collection_name,
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
    else:
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_FILE,
            job_id=redis_job_id,
            file_path=file_path,
            original_filename=original_filename,
            collection_name=collection_name,
            processing_config={"embedding_model_id": domain.embedding_model},
            document_id=document.id,
            db_url=settings.DATABASE_URL,
        )

    job.status = "pending"
    job.created_at = datetime.utcnow()
    job.completed_at = None
    job.error_message = None
    job.updated_at = datetime.utcnow()
    db.commit()
    logger.info("Job re-queued", job_id=job.id, redis_job_id=redis_job_id)
    return {"job_id": job.id, "message": "Job re-queued"}


@router.patch("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Mark a job as cancelled. For ingestion jobs, also updates Redis so worker can skip if not started."""
    job = _get_job_or_404(job_id, current_user, db)
    if job.status in ("completed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job already {job.status}",
        )

    job.status = "cancelled"
    payload = job.payload if isinstance(job.payload, dict) else {}
    redis_job_id = payload.get("redis_job_id")
    if redis_job_id and job.job_type.startswith("ingestion_"):
        try:
            redis_client = await get_redis_client()
            await redis_client.hset(f"job:{redis_job_id}", "status", "cancelled")
        except Exception as e:
            logger.warning(
                "Could not update Redis status for cancel", job_id=job_id, error=str(e)
            )
    db.commit()
    logger.info("Job cancelled", job_id=job.id)
    return {"job_id": job.id, "message": "Job cancelled"}


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a job record from the database. For ingestion jobs with document_id, updates the Document so the Data Domain UI reflects the removal."""
    job = _get_job_or_404(job_id, current_user, db)
    document_id = job.document_id
    is_ingestion = job.job_type.startswith("ingestion_")
    if document_id and is_ingestion:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            domain = (
                db.query(DataDomain)
                .filter(
                    DataDomain.id == doc.data_domain_id,
                    DataDomain.owner_id == current_user.id,
                )
                .first()
            )
            if domain:
                doc.status = "failed"
                doc.error_message = "Job removed by user"
                db.add(doc)
    db.delete(job)
    db.commit()
    logger.info(
        "Job deleted", job_id=job_id, user_id=current_user.id, document_id=document_id
    )
