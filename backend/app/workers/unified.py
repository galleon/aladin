"""
Unified ARQ worker: single queue, one entrypoint (run_task), delegates to handlers.

Run with: arq app.workers.unified.WorkerSettings

All jobs are enqueued as: enqueue_job('run_task', task_name='...', **kwargs)
Handlers implement TaskHandler and are registered at startup.
"""

import logging
import sys

import redis.asyncio as redis
from arq import run_worker
from arq.connections import RedisSettings

from app.config import settings
from app.arq_client import DEFAULT_QUEUE_NAME
from app.workers.base import run_task, register_handler, list_registered_tasks
from app.workers.transcription import TranscriptionHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    logger.info(
        "Starting unified worker; queue=%s redis=%s:%s db=%s",
        DEFAULT_QUEUE_NAME,
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

    # Register handlers that are always available (backend-only)
    register_handler(TranscriptionHandler())

    # Register ingestion handlers when pipeline package is on path (e.g. PYTHONPATH includes worker dir)
    try:
        from app.workers.handlers.ingestion import register_ingestion_handlers

        register_ingestion_handlers(register_handler)
    except ImportError as e:
        logger.info(
            "Ingestion handlers not registered (ingestion package not on path): %s",
            e,
        )

    logger.info("Registered tasks: %s", list_registered_tasks())


async def shutdown(ctx: dict) -> None:
    logger.info("Shutting down unified worker...")
    await ctx["redis"].aclose()


class WorkerSettings:
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        database=settings.REDIS_DB,
    )
    queue_name = DEFAULT_QUEUE_NAME
    functions = [run_task]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 2
    job_timeout = 3600
    max_tries = 1


if __name__ == "__main__":
    run_worker(WorkerSettings)
    sys.exit(0)
