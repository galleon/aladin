"""
Clip serving endpoint.

GET /api/clips/{collection}/{point_id}
  - Authenticated: requires active user
  - Fetches the Qdrant point for point_id in collection
  - Extracts clip_key from the payload
  - Returns a 307 redirect to a short-lived MinIO presigned URL (TTL = MINIO_PRESIGN_TTL_SECONDS)
  - Returns 404 when the point has no clip_key or the point does not exist
  - Returns 503 when MinIO is not configured or Qdrant is unreachable
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import settings
from ..models import User
from ..services.auth import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clips", tags=["clips"])


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )


def _get_minio_client():
    """Return a MinIO client or raise 503 if not configured."""
    if not settings.MINIO_ENDPOINT:
        raise HTTPException(
            status_code=503,
            detail="Clip storage is not configured on this server (MINIO_ENDPOINT not set).",
        )
    try:
        from minio import Minio

        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        return client
    except Exception as exc:
        logger.error("Failed to connect to MinIO: %s", exc)
        raise HTTPException(status_code=503, detail="Clip storage is unavailable.") from exc


@router.get(
    "/{collection}/{point_id}",
    summary="Redirect to a presigned URL for a video segment clip",
    response_class=RedirectResponse,
    responses={
        307: {"description": "Redirect to presigned MinIO URL"},
        404: {"description": "Point not found or has no clip"},
        503: {"description": "Clip storage not configured, unavailable, or Qdrant unreachable"},
    },
)
async def get_clip(
    collection: str,
    point_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Return a temporary redirect to the clip stored in MinIO for the given Qdrant point."""
    minio_client = _get_minio_client()
    qdrant = _get_qdrant_client()

    # Retrieve the Qdrant point — distinguish "not found" (404) from connection errors (503)
    try:
        results = qdrant.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
        )
    except UnexpectedResponse as exc:
        # 404 from Qdrant = collection or point genuinely absent
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection or point not found.") from exc
        logger.error("Qdrant error for %s/%s: %s", collection, point_id, exc)
        raise HTTPException(status_code=503, detail="Vector store returned an error.") from exc
    except Exception as exc:
        logger.error("Qdrant unreachable for %s/%s: %s", collection, point_id, exc)
        raise HTTPException(status_code=503, detail="Vector store is unavailable.") from exc

    if not results:
        raise HTTPException(status_code=404, detail="Point not found.")

    payload = results[0].payload or {}
    clip_key = payload.get("clip_key")
    if not clip_key:
        raise HTTPException(status_code=404, detail="No clip available for this point.")

    # Generate a presigned URL
    try:
        presign_ttl = timedelta(seconds=settings.MINIO_PRESIGN_TTL_SECONDS)
        url = minio_client.presigned_get_object(
            settings.MINIO_BUCKET,
            clip_key,
            expires=presign_ttl,
        )
    except Exception as exc:
        logger.error("Failed to generate presigned URL for %s: %s", clip_key, exc)
        raise HTTPException(status_code=503, detail="Could not generate clip URL.") from exc

    return RedirectResponse(url=url, status_code=307)
