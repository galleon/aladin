"""Avatar session API endpoint - creates a LiveKit room and enqueues the avatar worker."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import structlog

from ..database import get_db
from ..models import Agent, AgentType, User
from ..schemas import AvatarSessionResponse
from ..services.auth import get_current_active_user
from ..config import settings
from ..arq_client import get_arq_pool

logger = structlog.get_logger()

router = APIRouter(prefix="/agents", tags=["Avatar"])

try:
    from livekit.api import AccessToken, VideoGrants  # type: ignore[import]

    _LIVEKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    AccessToken = None  # type: ignore[assignment,misc]
    VideoGrants = None  # type: ignore[assignment,misc]
    _LIVEKIT_AVAILABLE = False


@router.post(
    "/{agent_id}/session",
    response_model=AvatarSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_avatar_session(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a LiveKit room for an avatar agent session.

    Enqueues an Arq task that launches the avatar worker, then returns
    the room name and a signed LiveKit token to the frontend.
    """
    # Validate agent exists and is accessible
    agent = (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            (Agent.owner_id == current_user.id) | (Agent.is_public == True),
        )
        .first()
    )
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )
    if agent.agent_type != AgentType.AVATAR.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session endpoint is only available for avatar agents",
        )

    # Generate a unique room name for this session
    room_name = f"avatar-{agent_id}-{uuid.uuid4().hex[:8]}"
    participant_identity = f"user-{current_user.id}"

    # Create a signed LiveKit access token
    try:
        if not _LIVEKIT_AVAILABLE or AccessToken is None:
            raise RuntimeError("livekit-api package is not installed")

        token = (
            AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_name(current_user.full_name or current_user.email)
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to create LiveKit token", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveKit token generation failed",
        ) from exc

    # Enqueue the avatar worker via Arq
    try:
        pool = await get_arq_pool()
        await pool.enqueue_job(
            "run_avatar_worker",
            agent_id=agent_id,
            room_name=room_name,
            livekit_url=settings.LIVEKIT_URL,
            livekit_api_key=settings.LIVEKIT_API_KEY,
            livekit_api_secret=settings.LIVEKIT_API_SECRET,
            avatar_config=agent.avatar_config or {},
            llm_model=agent.llm_model,
            system_prompt=agent.system_prompt,
            temperature=agent.temperature or 0.7,
            max_tokens=agent.max_tokens or 2048,
        )
        logger.info(
            "Enqueued avatar worker",
            agent_id=agent_id,
            room_name=room_name,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Failed to enqueue avatar worker (Redis may be unavailable)",
            error=str(exc),
        )
        # Continue — the token is still valid; the worker can be started manually

    return AvatarSessionResponse(
        room_name=room_name,
        token=token,
        livekit_url=settings.LIVEKIT_URL,
    )
