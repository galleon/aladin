"""
Worker interface and single-queue dispatcher.

All task types implement TaskHandler. One ARQ worker listens to a single queue
and delegates to the appropriate handler via run_task(task_name, **kwargs).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Task name constants (used when enqueueing and in the registry)
TASK_TRANSCRIPTION = "transcription"
TASK_INGESTION_WEB = "ingestion_web"
TASK_INGESTION_FILE = "ingestion_file"
TASK_INGESTION_VIDEO = "ingestion_video"


class TaskHandler(ABC):
    """Interface for background task handlers. All workers inherit from this."""

    @abstractmethod
    def task_name(self) -> str:
        """Return the task name this handler handles (e.g. 'transcription')."""
        pass

    @abstractmethod
    async def run(self, ctx: dict, **kwargs: Any) -> Any:
        """Execute the task. ctx contains shared worker context (e.g. redis)."""
        pass


_registry: Dict[str, TaskHandler] = {}


def register_handler(handler: TaskHandler) -> None:
    """Register a task handler. Called at worker startup."""
    name = handler.task_name()
    if name in _registry:
        logger.warning("Overwriting existing handler for task_name=%s", name)
    _registry[name] = handler
    logger.info("Registered task handler: %s", name)


def get_handler(task_name: str) -> Optional[TaskHandler]:
    """Return the handler for the given task name, or None."""
    return _registry.get(task_name)


def list_registered_tasks() -> list[str]:
    """Return all registered task names (for logging/debug)."""
    return list(_registry.keys())


async def run_task(ctx: dict, task_name: str, **kwargs: Any) -> Any:
    """
    Single ARQ job entrypoint: delegate to the right handler by task_name.

    All jobs are enqueued as:
        enqueue_job('run_task', task_name='transcription', transcription_job_id=123)
        enqueue_job('run_task', task_name='ingestion_web', job_id=..., url=..., ...)

    If an ingestion job fails (including before the handler updates Redis), we set
    job:{job_id} status to "failed" in Redis so the Jobs UI shows Failed instead of Queued.
    """
    job_id = ctx.get("job_id", "?")
    logger.info("Job picked from queue job_id=%s task_name=%s", job_id, task_name)
    handler = get_handler(task_name)
    if handler is None:
        known = list_registered_tasks()
        raise ValueError(f"Unknown task_name={task_name!r}. Registered: {known}")
    logger.info("Dispatching task task_name=%s", task_name)
    try:
        return await handler.run(ctx, **kwargs)
    except Exception as e:
        # So the Jobs UI shows Failed instead of stuck Queued when the worker errors
        if task_name.startswith("ingestion_"):
            redis_job_id = kwargs.get("job_id")
            if redis_job_id and isinstance(redis_job_id, str):
                redis_client = ctx.get("redis")
                if redis_client:
                    try:
                        await redis_client.hset(
                            f"job:{redis_job_id}", "status", "failed"
                        )
                        await redis_client.hset(
                            f"job:{redis_job_id}",
                            "message",
                            (str(e)[:500] if str(e) else "Job failed"),
                        )
                    except Exception as redis_err:
                        logger.warning(
                            "Could not update Redis job status on failure: %s",
                            redis_err,
                        )
        raise
