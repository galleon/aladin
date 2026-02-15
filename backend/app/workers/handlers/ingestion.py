"""
Ingestion task handlers. Lazy-import from ingestion worker when available.

When the unified worker runs with the pipeline on PYTHONPATH (e.g. worker
dir mounted at /app/worker), these handlers delegate to the pipeline's process_* functions.
"""

import logging
from typing import Any

from app.workers.base import (
    TaskHandler,
    TASK_INGESTION_WEB,
    TASK_INGESTION_FILE,
    TASK_INGESTION_VIDEO,
)

logger = logging.getLogger(__name__)


def _get_ingestion_functions():
    """Lazy import so ingestion is optional. Raises ImportError if not on path."""
    from worker.main import process_web_job, process_file_job, process_video_job

    return process_web_job, process_file_job, process_video_job


def _kwargs_without_task_name(kwargs: dict) -> dict:
    """Remove task_name so ingestion process_* functions don't get an extra arg."""
    return {k: v for k, v in kwargs.items() if k != "task_name"}


class IngestionWebHandler(TaskHandler):
    def task_name(self) -> str:
        return TASK_INGESTION_WEB

    async def run(self, ctx: dict, **kwargs: Any) -> Any:
        process_web_job, _, _ = _get_ingestion_functions()
        return await process_web_job(ctx, **_kwargs_without_task_name(kwargs))


class IngestionFileHandler(TaskHandler):
    def task_name(self) -> str:
        return TASK_INGESTION_FILE

    async def run(self, ctx: dict, **kwargs: Any) -> Any:
        _, process_file_job, _ = _get_ingestion_functions()
        return await process_file_job(ctx, **_kwargs_without_task_name(kwargs))


class IngestionVideoHandler(TaskHandler):
    def task_name(self) -> str:
        return TASK_INGESTION_VIDEO

    async def run(self, ctx: dict, **kwargs: Any) -> Any:
        _, _, process_video_job = _get_ingestion_functions()
        return await process_video_job(ctx, **_kwargs_without_task_name(kwargs))


def register_ingestion_handlers(register_handler_fn) -> None:
    """Register ingestion handlers. Call only if ingestion package is available."""
    register_handler_fn(IngestionWebHandler())
    register_handler_fn(IngestionFileHandler())
    register_handler_fn(IngestionVideoHandler())
    logger.info("Registered ingestion handlers: web, file, video")
