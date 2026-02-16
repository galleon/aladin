"""API endpoints for model discovery."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..services.auth import get_current_user
from ..services.model_service import model_service, ModelInfo
from ..models import User

router = APIRouter(prefix="/models", tags=["models"])


class ModelsResponse(BaseModel):
    """Response containing list of available models."""

    models: list[ModelInfo]
    endpoint: str
    error: str | None = None


class EndpointConfig(BaseModel):
    """Current endpoint configuration (for debugging)."""

    llm_base: str
    embedding_base: str


@router.get("/llm", response_model=ModelsResponse)
async def list_llm_models(
    refresh: bool = False,
    current_user: User = Depends(get_current_user),
):
    """List available LLM models from the configured endpoint."""
    from ..config import settings

    models = await model_service.get_llm_models(refresh=refresh)
    error = model_service.get_last_llm_error() if not models else None
    return ModelsResponse(
        models=models, endpoint=settings.LLM_API_BASE, error=error
    )


@router.get("/embedding", response_model=ModelsResponse)
async def list_embedding_models(
    refresh: bool = False,
    current_user: User = Depends(get_current_user),
):
    """List available embedding models from the configured endpoint."""
    from ..config import settings

    models = await model_service.get_embedding_models(refresh=refresh)
    return ModelsResponse(models=models, endpoint=settings.EMBEDDING_API_BASE)


@router.get("/config", response_model=EndpointConfig)
async def get_endpoint_config(
    current_user: User = Depends(get_current_user),
):
    """Get current endpoint configuration."""
    from ..config import settings

    return EndpointConfig(
        llm_base=settings.LLM_API_BASE,
        embedding_base=settings.EMBEDDING_API_BASE,
    )


@router.post("/refresh")
async def refresh_model_cache(
    current_user: User = Depends(get_current_user),
):
    """Clear model cache and refresh from endpoints."""
    model_service.clear_cache()
    llm_models = await model_service.get_llm_models(refresh=True)
    embedding_models = await model_service.get_embedding_models(refresh=True)
    return {
        "message": "Model cache refreshed",
        "llm_count": len(llm_models),
        "embedding_count": len(embedding_models),
    }


@router.get("/check/{model_name}")
async def check_model_availability(
    model_name: str,
    current_user: User = Depends(get_current_user),
):
    """Check if a specific model is available."""
    from ..config import settings

    llm_models = await model_service.get_llm_models(refresh=False)
    embedding_models = await model_service.get_embedding_models(refresh=False)

    all_models = [m.id for m in llm_models + embedding_models]
    is_available = model_name in all_models

    return {
        "model": model_name,
        "available": is_available,
        "endpoint": settings.LLM_API_BASE
        if model_name in [m.id for m in llm_models]
        else settings.EMBEDDING_API_BASE,
    }
