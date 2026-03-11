"""
Clip serving endpoint.

GET /api/clips/{collection}/{point_id}
  - Authenticated: requires active user
  - Fetches the Qdrant point for point_id in collection
  - Extracts clip_key from the payload
  - Returns a 307 redirect to a short-lived MinIO presigned URL (TTL = MINIO_PRESIGN_TTL_SECONDS)
  - Returns 404 when the point has no clip_key or the point does not exist
  - Returns 503 when MinIO is not configured or Qdrant is unreachable

GET /api/clips/{collection}/{point_id}/url
  - Same auth + lookup, but returns {"url": "<presigned>"} as JSON instead of a redirect.
  - Used by the frontend to obtain the presigned URL before setting it as a <video> src,
    since <video src> cannot send the JWT Authorization header.
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


def _get_minio_client(public: bool = False):
    """Return a MinIO client or raise 503 if not configured.

    When public=True, use MINIO_PUBLIC_ENDPOINT (falling back to MINIO_ENDPOINT) so that
    presigned URLs are routable from browsers rather than pointing at the internal Docker hostname.
    """
    if not settings.MINIO_ENDPOINT:
        raise HTTPException(
            status_code=503,
            detail="Clip storage is not configured on this server (MINIO_ENDPOINT not set).",
        )
    endpoint = (settings.MINIO_PUBLIC_ENDPOINT or settings.MINIO_ENDPOINT) if public else settings.MINIO_ENDPOINT
    try:
        from minio import Minio

        client = Minio(
            endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        return client
    except Exception as exc:
        logger.error("Failed to connect to MinIO: %s", exc)
        raise HTTPException(status_code=503, detail="Clip storage is unavailable.") from exc


def _resolve_clip_key(collection: str, point_id: str) -> str:
    """Look up the clip_key for a Qdrant point. Raises 404/503 as appropriate."""
    qdrant = _get_qdrant_client()
    try:
        results = qdrant.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
        )
    except UnexpectedResponse as exc:
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
    return clip_key


def _presign(clip_key: str) -> str:
    """Generate a presigned URL using the public MinIO endpoint."""
    minio_client = _get_minio_client(public=True)
    try:
        presign_ttl = timedelta(seconds=settings.MINIO_PRESIGN_TTL_SECONDS)
        return minio_client.presigned_get_object(
            settings.MINIO_BUCKET,
            clip_key,
            expires=presign_ttl,
        )
    except Exception as exc:
        logger.error("Failed to generate presigned URL for %s: %s", clip_key, exc)
        raise HTTPException(status_code=503, detail="Could not generate clip URL.") from exc


@router.get(
    "/{collection}/{point_id}/url",
    summary="Return the presigned URL for a video segment clip as JSON",
    responses={
        200: {"description": "JSON object with presigned URL", "content": {"application/json": {"example": {"url": "http://..."}}}},
        404: {"description": "Point not found or has no clip"},
        503: {"description": "Clip storage not configured, unavailable, or Qdrant unreachable"},
    },
)
async def get_clip_url(
    collection: str,
    point_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Return {"url": "<presigned>"} so the frontend can set it as a <video> src."""
    clip_key = _resolve_clip_key(collection, point_id)
    url = _presign(clip_key)
    return {"url": url}


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
    clip_key = _resolve_clip_key(collection, point_id)
    url = _presign(clip_key)
    return RedirectResponse(url=url, status_code=307)
