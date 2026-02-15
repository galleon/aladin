"""Video transcription API endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
import structlog
import uuid

from ..database import get_db
from ..models import User, Agent, AgentType, TranscriptionJob
from ..services.auth import get_current_active_user
from ..config import settings
from ..arq_client import (
    get_arq_pool,
    get_redis_client,
    DEFAULT_QUEUE_NAME,
    KNOWN_QUEUES,
)
from ..workers.base import TASK_TRANSCRIPTION
from ..schemas import (
    TranscriptionJobListResponse,
    TranscriptionJobDetailResponse,
    TranscriptionJobQueuedResponse,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/video-transcription", tags=["Video Transcription"])


def check_whisper_api_configured():
    """Check if Whisper API is configured, raise error if not."""
    if not settings.WHISPER_API_BASE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video transcription is not available. WHISPER_API_BASE must be configured.",
        )


def _verify_agent_and_access(agent_id: int, current_user: User, db: Session) -> Agent:
    """Verify agent exists, is video_transcription type, and user has access."""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    if agent.agent_type != AgentType.VIDEO_TRANSCRIPTION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not a video transcription agent",
        )
    if not agent.is_public and agent.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this agent",
        )
    return agent


@router.post(
    "/transcribe",
    response_model=TranscriptionJobQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def transcribe_video(
    agent_id: int,
    video: UploadFile = File(...),
    language: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Queue a video transcription job. Returns immediately with job_id.
    When the job completes, you can fetch the result via GET /video-transcription/jobs/{job_id}.
    """
    check_whisper_api_configured()
    _verify_agent_and_access(agent_id, current_user, db)

    video_content = await video.read()
    video_size = len(video_content)
    if video_size > settings.MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file too large. Maximum size: {settings.MAX_VIDEO_SIZE / 1024 / 1024}MB",
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    job_uuid = f"{timestamp}_{str(uuid.uuid4())[:8]}"
    job_dir = (
        Path(settings.UPLOAD_DIR)
        / "jobs"
        / "transcription"
        / str(current_user.id)
        / job_uuid
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    source_filename = video.filename or "video.mp4"
    source_path = job_dir / f"source_{source_filename}"
    source_path.write_bytes(video_content)

    job_directory_rel = f"jobs/transcription/{current_user.id}/{job_uuid}"
    job = TranscriptionJob(
        user_id=current_user.id,
        agent_id=agent_id,
        job_type="transcribe",
        status="pending",
        source_filename=source_filename,
        source_path=str(source_path),
        job_directory=job_directory_rel,
        language=language,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        arq_pool = await get_arq_pool()
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_TRANSCRIPTION,
            transcription_job_id=job.id,
        )
    except Exception as e:
        logger.error("Failed to enqueue transcription job", job_id=job.id, error=str(e))
        job.status = "failed"
        job.error_message = f"Failed to enqueue: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue job: {str(e)}",
        )

    logger.info("Transcription job queued", job_id=job.id, filename=source_filename)
    return TranscriptionJobQueuedResponse(job_id=job.id)


@router.post(
    "/add-subtitles",
    response_model=TranscriptionJobQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def add_subtitles_to_video(
    agent_id: int,
    video: UploadFile = File(...),
    language: str | None = None,
    subtitle_language: str | None = None,
    translation_agent_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Queue a video transcription + subtitles job. Returns immediately with job_id.
    When the job completes, fetch the result via GET /video-transcription/jobs/{job_id} and download via GET .../jobs/{job_id}/download.
    """
    check_whisper_api_configured()
    _verify_agent_and_access(agent_id, current_user, db)

    video_content = await video.read()
    video_size = len(video_content)
    if video_size > settings.MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file too large. Maximum size: {settings.MAX_VIDEO_SIZE / 1024 / 1024}MB",
        )

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    job_uuid = f"{timestamp}_{str(uuid.uuid4())[:8]}"
    job_dir = (
        Path(settings.UPLOAD_DIR)
        / "jobs"
        / "transcription"
        / str(current_user.id)
        / job_uuid
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    source_filename = video.filename or "video.mp4"
    source_path = job_dir / f"source_{source_filename}"
    source_path.write_bytes(video_content)

    job_directory_rel = f"jobs/transcription/{current_user.id}/{job_uuid}"
    job = TranscriptionJob(
        user_id=current_user.id,
        agent_id=agent_id,
        job_type="add_subtitles",
        status="pending",
        source_filename=source_filename,
        source_path=str(source_path),
        job_directory=job_directory_rel,
        language=language,
        subtitle_language=subtitle_language,
        translation_agent_id=translation_agent_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        arq_pool = await get_arq_pool()
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_TRANSCRIPTION,
            transcription_job_id=job.id,
        )
    except Exception as e:
        logger.error("Failed to enqueue add-subtitles job", job_id=job.id, error=str(e))
        job.status = "failed"
        job.error_message = f"Failed to enqueue: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue job: {str(e)}",
        )

    logger.info("Add-subtitles job queued", job_id=job.id, filename=source_filename)
    return TranscriptionJobQueuedResponse(job_id=job.id)


QUEUE_STATUS_TIMEOUT = 2.0  # seconds; avoid blocking UI if Redis is slow/unreachable


@router.get("/queue-status")
async def get_queue_status(
    current_user: User = Depends(get_current_active_user),
):
    """Number of jobs waiting in the job queue (arq:queue)."""
    try:
        redis_client = await get_redis_client()
        queued = await asyncio.wait_for(
            redis_client.zcard(DEFAULT_QUEUE_NAME),
            timeout=QUEUE_STATUS_TIMEOUT,
        )
        return {"queue_name": DEFAULT_QUEUE_NAME, "queued_count": queued}
    except asyncio.TimeoutError:
        logger.warning("Queue status timed out", timeout=QUEUE_STATUS_TIMEOUT)
        return {"queue_name": DEFAULT_QUEUE_NAME, "queued_count": None, "error": "timeout"}
    except Exception as e:
        logger.warning("Queue status check failed", error=str(e))
        return {"queue_name": DEFAULT_QUEUE_NAME, "queued_count": None, "error": str(e)}


@router.get("/queue-debug")
async def get_queue_debug(
    current_user: User = Depends(get_current_active_user),
):
    """
    Debug: queue count, sample job IDs in Redis, and Redis connection hint.
    Use this when jobs stay 'queued' to verify backend and worker use the same Redis.
    """
    try:
        redis_client = await get_redis_client()
        queued = await asyncio.wait_for(
            redis_client.zcard(DEFAULT_QUEUE_NAME),
            timeout=QUEUE_STATUS_TIMEOUT,
        )
        job_ids_sample = await asyncio.wait_for(
            redis_client.zrange(DEFAULT_QUEUE_NAME, 0, 9),
            timeout=QUEUE_STATUS_TIMEOUT,
        )
        return {
            "queue_name": DEFAULT_QUEUE_NAME,
            "queued_count": queued,
            "job_ids_sample": list(job_ids_sample) if job_ids_sample else [],
            "why_pending_in_ui_but_zero_here": (
                "The job list in the UI comes from the database (status=pending). "
                "The number here is Redis (arq:queue). If you see pending in the UI but 0 here, enqueue may have failed or jobs were lost."
            ),
            "redis_hint": {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB,
            },
            "worker_checks": [
                "Service name must be 'worker'. Run: docker compose ps worker",
                "Worker must listen to arq:queue. Run: docker compose exec worker python3 -c \"from app.workers.unified import WorkerSettings; print('queue:', getattr(WorkerSettings, 'queue_name'))\"",
                "If queue is wrong or worker missing, rebuild and start: docker compose up -d --build worker",
                "Logs: docker compose logs worker (look for 'Starting unified worker; queue=arq:queue').",
            ],
        }
    except asyncio.TimeoutError:
        return {"error": "timeout", "queue_name": DEFAULT_QUEUE_NAME}
    except Exception as e:
        logger.warning("Queue debug failed", error=str(e))
        return {"error": str(e), "queue_name": DEFAULT_QUEUE_NAME}


@router.get("/jobs", response_model=list[TranscriptionJobListResponse])
def list_transcription_jobs(
    agent_id: int | None = Query(None, description="Filter by agent"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List current user's transcription jobs, most recent first. Omit agent_id to see all jobs."""
    q = db.query(TranscriptionJob).filter(TranscriptionJob.user_id == current_user.id)
    if agent_id is not None:
        q = q.filter(TranscriptionJob.agent_id == agent_id)
    jobs = q.order_by(TranscriptionJob.created_at.desc()).all()
    return [
        TranscriptionJobListResponse(
            id=j.id,
            agent_id=j.agent_id,
            job_type=j.job_type,
            status=j.status,
            source_filename=j.source_filename,
            created_at=j.created_at,
            completed_at=j.completed_at,
        )
        for j in jobs
    ]


def _get_job_or_404(job_id: int, current_user: User, db: Session) -> TranscriptionJob:
    job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    return job


@router.post("/jobs/{job_id}/requeue", status_code=status.HTTP_200_OK)
async def requeue_transcription_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Re-queue a pending job (e.g. when it was never picked or enqueue failed). Only pending jobs can be re-queued."""
    job = _get_job_or_404(job_id, current_user, db)
    if job.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only pending jobs can be re-queued; current status is {job.status!r}",
        )
    try:
        arq_pool = await get_arq_pool()
        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_TRANSCRIPTION,
            transcription_job_id=job.id,
        )
    except Exception as e:
        logger.error("Failed to re-queue job", job_id=job.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-queue: {str(e)}",
        )
    logger.info("Transcription job re-queued", job_id=job.id)
    return {"job_id": job.id, "message": "Job re-queued"}


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transcription_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a single transcription job (DB only; queue entry may still exist)."""
    job = _get_job_or_404(job_id, current_user, db)
    db.delete(job)
    db.commit()
    logger.info("Transcription job deleted", job_id=job_id, user_id=current_user.id)
    return None


@router.delete("/jobs", status_code=status.HTTP_200_OK)
def purge_user_transcription_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete all transcription jobs for the current user."""
    deleted = db.query(TranscriptionJob).filter(TranscriptionJob.user_id == current_user.id).delete()
    db.commit()
    logger.info("User transcription jobs purged", user_id=current_user.id, deleted=deleted)
    return {"deleted": deleted}


@router.get("/queues")
async def list_queues(
    current_user: User = Depends(get_current_active_user),
):
    """List Redis job queues with counts (transcription, ingestion)."""
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
            result.append({"name": queue_name, "label": label, "count": count})
        return {"queues": result}
    except Exception as e:
        logger.warning("List queues failed", error=str(e))
        return {"queues": [], "error": str(e)}


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
        logger.info("Queue purged", queue_name=queue_name, removed=removed, user_id=current_user.id)
        return {"queue_name": queue_name, "removed": removed}
    except Exception as e:
        logger.warning("Purge queue failed", queue_name=queue_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/jobs/{job_id}", response_model=TranscriptionJobDetailResponse)
def get_transcription_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get transcription job detail including result when completed."""
    job = _get_job_or_404(job_id, current_user, db)
    return TranscriptionJobDetailResponse(
        id=job.id,
        agent_id=job.agent_id,
        job_type=job.job_type,
        status=job.status,
        source_filename=job.source_filename,
        language=job.language,
        subtitle_language=job.subtitle_language,
        error_message=job.error_message,
        result_transcript=job.result_transcript,
        result_video_path=job.result_video_path,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs/{job_id}/result")
def get_transcription_result(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get transcript JSON for a completed job."""
    job = _get_job_or_404(job_id, current_user, db)
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job not completed (status: {job.status})",
        )
    if not job.result_transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcript result for this job",
        )
    return JSONResponse(content=job.result_transcript)


@router.get("/jobs/{job_id}/download")
def download_transcription_video(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Download the video with embedded subtitles (add_subtitles jobs only)."""
    job = _get_job_or_404(job_id, current_user, db)
    if job.job_type != "add_subtitles":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Download is only available for add_subtitles jobs",
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job not completed (status: {job.status})",
        )
    if not job.result_video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No video result for this job",
        )
    path = Path(job.result_video_path)
    if not path.is_absolute():
        path = Path(settings.UPLOAD_DIR) / job.result_video_path
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found",
        )
    download_filename = f"{Path(job.source_filename).stem}_with_subtitles.mp4"
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=download_filename,
        headers={"Content-Disposition": f'attachment; filename="{download_filename}"'},
    )
