"""
ARQ worker for video transcription jobs.
Run with: arq app.workers.transcription.WorkerSettings
"""

import asyncio
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from arq.connections import RedisSettings

from app.config import settings
# Standalone worker only (prefer unified worker). Main app uses single queue arq:queue.
_STANDALONE_QUEUE = "arq:transcription"
from app.database import SessionLocal
from app.workers.base import TaskHandler, TASK_TRANSCRIPTION, register_handler
from app.models import TranscriptionJob, Agent, AgentType
from app.services.video_transcription_service import video_transcription_service
from app.services.email_service import send_transcription_ready_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_transcription_pipeline(transcription_job_id: int) -> None:
    """
    Sync pipeline: load job, transcribe (and optionally translate + add subtitles), update DB, send email.
    Runs in a thread from the ARQ worker.
    """
    try:
        job_id_int = int(transcription_job_id)
    except (TypeError, ValueError):
        logger.error("Invalid transcription_job_id: %s", transcription_job_id)
        return
    db = None
    try:
        db = SessionLocal()
        job = (
            db.query(TranscriptionJob)
            .filter(TranscriptionJob.id == job_id_int)
            .first()
        )
        if not job:
            logger.error("Transcription job not found job_id=%s", job_id_int)
            return
        if job.status not in ("pending", "processing"):
            logger.warning(
                "Transcription job already processed job_id=%s status=%s",
                job_id_int,
                job.status,
            )
            return

        logger.info(
            "Processing transcription job job_id=%s job_type=%s source_filename=%s user_id=%s",
            job.id,
            job.job_type,
            job.source_filename,
            job.user_id,
        )
        job.status = "processing"
        db.commit()

        agent = db.query(Agent).filter(Agent.id == job.agent_id).first()
        if not agent:
            logger.error("Transcription job failed: agent not found job_id=%s agent_id=%s", job.id, job.agent_id)
            job.status = "failed"
            job.error_message = "Agent not found"
            db.commit()
            return
        whisper_model_size = getattr(agent, "whisper_model_size", None) or "base"
        job_dir = Path(settings.UPLOAD_DIR) / job.job_directory
        job_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(job.source_path)
        if not source_path.is_absolute():
            source_path = job_dir / Path(job.source_path).name
        if not source_path.exists():
            source_path = job_dir / f"source_{job.source_filename}"
        if not source_path.exists():
            logger.error(
                "Transcription job failed: source file not found job_id=%s source_path=%s job_directory=%s",
                job.id,
                source_path,
                job.job_directory,
            )
            job.status = "failed"
            job.error_message = f"Source file not found: {source_path}"
            db.commit()
            return

        try:
            with open(source_path, "rb") as video_file:
                result = video_transcription_service.transcribe_video(
                    video_file, language=job.language, model_size=whisper_model_size
                )
        except Exception as e:
            logger.exception(
                "Transcription failed job_id=%s error=%s source_path=%s",
                job.id,
                e,
                source_path,
            )
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
            return

        transcript_payload = {
            "transcript": result["transcript"],
            "segments": result["segments"],
            "language": result["language"],
            "language_probability": result.get("language_probability", 1.0),
        }
        segments = result["segments"]
        original_srt_path = job_dir / "original.srt"
        video_transcription_service.save_srt_file(segments, str(original_srt_path))

        if job.job_type == "transcribe":
            job.result_transcript = transcript_payload
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            db.commit()
            user = job.user
            if user and user.email:
                send_transcription_ready_email(
                    to_email=user.email,
                    job_id=job.id,
                    agent_id=job.agent_id,
                    source_filename=job.source_filename,
                )
            logger.info("Transcription job completed job_id=%s", job.id)
            return

        # add_subtitles: translate if needed, then add subtitles to video
        translated_srt_path = job_dir / "translated.srt"
        output_video_path = job_dir / "final_video.mp4"
        if (
            job.subtitle_language
            and job.subtitle_language.lower() != result["language"].lower()
        ):
            translation_agent = None
            if job.translation_agent_id:
                translation_agent = (
                    db.query(Agent)
                    .filter(
                        Agent.id == job.translation_agent_id,
                        Agent.agent_type == AgentType.TRANSLATION,
                    )
                    .first()
                )
            if not translation_agent:
                translation_agent = (
                    db.query(Agent)
                    .filter(
                        Agent.agent_type == AgentType.TRANSLATION,
                        Agent.owner_id == job.user_id,
                    )
                    .first()
                )
            if translation_agent:
                try:
                    segments = video_transcription_service.translate_subtitles(
                        segments,
                        result["language"],
                        job.subtitle_language,
                        translation_agent=translation_agent,
                    )
                    video_transcription_service.save_srt_file(
                        segments, str(translated_srt_path)
                    )
                except Exception as e:
                    logger.warning(
                        "Subtitle translation failed, using original: %s", e
                    )
                    shutil.copy(original_srt_path, translated_srt_path)
            else:
                shutil.copy(original_srt_path, translated_srt_path)
        else:
            shutil.copy(original_srt_path, translated_srt_path)

        try:
            video_transcription_service.add_subtitles_to_video(
                str(source_path),
                segments,
                str(output_video_path),
            )
        except Exception as e:
            logger.exception("Add subtitles failed job_id=%s", job.id)
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
            return

        job.result_transcript = transcript_payload
        job.result_video_path = f"{job.job_directory}/final_video.mp4"
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        user = job.user
        if user and user.email:
            send_transcription_ready_email(
                to_email=user.email,
                job_id=job.id,
                agent_id=job.agent_id,
                source_filename=job.source_filename,
            )
        logger.info("Add-subtitles job completed job_id=%s", job.id)
    except Exception as e:
        logger.exception("Transcription pipeline failed job_id=%s error=%s", job_id_int, e)
        try:
            fail_db = SessionLocal()
            job = fail_db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id_int).first()
            if job and job.status in ("pending", "processing"):
                job.status = "failed"
                job.error_message = str(e)[:2000]
                job.completed_at = datetime.utcnow()
                fail_db.commit()
                logger.info("Marked job as failed job_id=%s", job_id_int)
            fail_db.close()
        except Exception as commit_err:
            logger.exception("Could not mark job as failed job_id=%s error=%s", job_id_int, commit_err)
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


async def process_transcription_job(ctx: dict, transcription_job_id: int):
    """ARQ job: run the sync transcription pipeline in a thread (legacy entrypoint)."""
    logger.info("Transcription job picked from queue job_id=%s", transcription_job_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_transcription_pipeline, transcription_job_id)


class TranscriptionHandler(TaskHandler):
    """Handler for transcription tasks; used by the unified worker dispatcher."""

    def task_name(self) -> str:
        return TASK_TRANSCRIPTION

    async def run(self, ctx: dict, **kwargs: Any) -> None:
        raw_id = kwargs.get("transcription_job_id")
        if raw_id is None:
            raise ValueError("transcription_job_id is required")
        try:
            transcription_job_id = int(raw_id)
        except (TypeError, ValueError):
            raise ValueError(f"transcription_job_id must be an integer, got {type(raw_id).__name__!r}")
        logger.info("Transcription job picked from queue job_id=%s", transcription_job_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_transcription_pipeline, transcription_job_id)


async def startup(ctx: dict):
    logger.info("Starting Transcription Worker...")
    logger.info(
        "Listening to queue=%s redis=%s:%s db=%s (standalone worker; prefer unified)",
        _STANDALONE_QUEUE,
        settings.REDIS_HOST,
        settings.REDIS_PORT,
        settings.REDIS_DB,
    )
    ctx["redis"] = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True,
    )


async def shutdown(ctx: dict):
    logger.info("Shutting down Transcription Worker...")
    await ctx["redis"].aclose()


class WorkerSettings:
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        database=settings.REDIS_DB,
    )
    queue_name = _STANDALONE_QUEUE
    functions = [process_transcription_job]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 2
    job_timeout = 3600  # 1 hour for long videos
    max_tries = 1


if __name__ == "__main__":
    from arq import run_worker

    run_worker(WorkerSettings)
