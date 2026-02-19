"""ingest_url tool – enqueues a web crawling ARQ job for ingestion."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger()


@tool
def ingest_url(
    url: str,
    collection_name: str,
    depth_limit: int = 2,
    max_pages: int = 100,
) -> str:
    """Trigger web crawling and ingestion of a URL into the vector store.

    Args:
        url: The starting URL to crawl.
        collection_name: Qdrant collection to store vectors in.
        depth_limit: How many link-levels deep to crawl (0-10, default 2).
        max_pages: Maximum pages to process (1-1000, default 100).

    Returns:
        A message with the queued job ID, or an error string.
    """
    if not url or not collection_name:
        return "Error: url and collection_name are required."

    async def _enqueue() -> str:
        from ..arq_client import get_arq_pool, get_redis_client
        from ..workers.base import TASK_INGESTION_WEB

        job_id = f"web_{uuid.uuid4().hex[:12]}"

        arq_pool = await get_arq_pool()
        redis_client = await get_redis_client()

        source_config = {
            "url": url,
            "depth_limit": max(0, min(depth_limit, 10)),
            "strategy": "bfs",
            "inclusion_patterns": [],
            "exclusion_patterns": [],
            "max_pages": max(1, min(max_pages, 1000)),
        }
        processing_config = {
            "render_js": False,
            "wait_for_selector": None,
            "wait_timeout": 30,
            "extract_tables": True,
            "table_strategy": "markdown",
            "extract_images": False,
            "vlm_model": None,
            "extract_links": True,
        }

        job_data = {
            "job_id": job_id,
            "status": "queued",
            "type": "web",
            "url": url,
            "collection_name": collection_name,
            "created_at": datetime.utcnow().isoformat(),
            "progress": "0",
            "pages_processed": "0",
            "pages_total": "0",
            "chunks_created": "0",
        }
        await redis_client.hset(f"job:{job_id}", mapping=job_data)
        await redis_client.expire(f"job:{job_id}", 86400)

        await arq_pool.enqueue_job(
            "run_task",
            task_name=TASK_INGESTION_WEB,
            job_id=job_id,
            url=url,
            collection_name=collection_name,
            source_config=source_config,
            processing_config=processing_config,
            metadata={},
        )
        return job_id

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                job_id = pool.submit(asyncio.run, _enqueue()).result()
        else:
            job_id = asyncio.run(_enqueue())

        logger.info("ingest_url tool queued job", job_id=job_id, url=url)
        return f"Web ingestion job queued: {job_id}"

    except Exception as e:
        logger.error("ingest_url tool failed", error=str(e))
        return f"Ingestion failed: {e}"
