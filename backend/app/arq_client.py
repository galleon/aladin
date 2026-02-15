"""Shared ARQ and Redis clients for enqueueing background jobs."""

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings

from .config import settings

_arq_pool = None
_redis_client = None

# Single queue for all background tasks (transcription, ingestion). Unified worker listens here.
DEFAULT_QUEUE_NAME = "arq:queue"

KNOWN_QUEUES = [(DEFAULT_QUEUE_NAME, "Job queue")]


async def get_arq_pool():
    """Get or create ARQ connection pool (single queue for all tasks)."""
    global _arq_pool
    if _arq_pool is None:
        redis_settings = RedisSettings(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            database=settings.REDIS_DB,
        )
        _arq_pool = await create_pool(
            redis_settings,
            default_queue_name=DEFAULT_QUEUE_NAME,
        )
    return _arq_pool


async def get_redis_client():
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=3,
        )
    return _redis_client
