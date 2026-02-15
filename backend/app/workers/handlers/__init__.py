"""Task handlers for the unified worker. Each handler implements TaskHandler."""

from app.workers.base import (
    TASK_TRANSCRIPTION,
    TASK_INGESTION_WEB,
    TASK_INGESTION_FILE,
    TASK_INGESTION_VIDEO,
)

__all__ = [
    "TASK_TRANSCRIPTION",
    "TASK_INGESTION_WEB",
    "TASK_INGESTION_FILE",
    "TASK_INGESTION_VIDEO",
]
