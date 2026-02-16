"""Service for discovering available models from API endpoints."""

from __future__ import annotations

import httpx
import structlog
from pydantic import BaseModel

from ..config import settings

logger = structlog.get_logger()


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    name: str
    owned_by: str | None = None
    type: str  # "llm" or "embedding"


class ModelService:
    """Service for discovering models from OpenAI-compatible endpoints."""

    def __init__(self):
        self._llm_models_cache: list[ModelInfo] | None = None
        self._embedding_models_cache: list[ModelInfo] | None = None
        self._llm_fetch_error: str | None = None
        self._embedding_fetch_error: str | None = None

    def _is_placeholder_key(self, api_key: str | None) -> bool:
        """Treat empty or common placeholder keys as 'no auth required'."""
        if not api_key or not api_key.strip():
            return True
        lower = api_key.strip().lower()
        return lower in ("sk-dummy-key", "dummy", "placeholder", "none")

    async def _fetch_models(
        self, base_url: str, api_key: str
    ) -> tuple[list[dict], str | None]:
        """Fetch models from an OpenAI-compatible /v1/models endpoint.
        Returns (list of model dicts, error message or None).
        """
        url = f"{base_url.rstrip('/')}/models"
        use_auth = not self._is_placeholder_key(api_key)
        headers = (
            {"Authorization": f"Bearer {api_key.strip()}"} if use_auth else {}
        )

        async def do_request(hdrs: dict) -> tuple[list[dict], str | None]:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, headers=hdrs)
                    response.raise_for_status()
                    data = response.json()
                    return (data.get("data", []), None)
            except httpx.TimeoutException:
                return ([], "Request timed out")
            except httpx.HTTPStatusError as e:
                return ([], f"HTTP {e.response.status_code}")
            except httpx.ConnectError as e:
                return ([], str(e) or "Connection failed")
            except Exception as e:
                return ([], str(e))

        result, err = await do_request(headers)
        if err and use_auth and "401" in err:
            # Retry without auth in case the server does not require it
            result, err = await do_request({})
        return (result, err)

    async def get_llm_models(self, refresh: bool = False) -> list[ModelInfo]:
        """Get available LLM models from the configured endpoint."""
        if self._llm_models_cache is not None and not refresh:
            return self._llm_models_cache

        self._llm_fetch_error = None
        models_data, err = await self._fetch_models(
            settings.LLM_API_BASE,
            settings.LLM_API_KEY,
        )
        if err:
            self._llm_fetch_error = err
            logger.warning(
                "LLM models fetch failed",
                endpoint=settings.LLM_API_BASE,
                error=err,
            )
            return []

        # Exclude embedding, docling, reranker and similar non-LLM models
        # (they should only be chosen in Data Domain / ingestion pipeline)
        _EXCLUDE_PATTERNS = (
            "embedding", "embed-", "bge-", "e5-", "nomic-embed", "text-embedding",
            "docling", "granite-docling",
            "rerank", "reranker", "bge-reranker", "bge-reranker-",
        )

        models = []
        for model in models_data:
            model_id = model.get("id", "") or ""
            lower = model_id.lower()
            if any(p in lower for p in _EXCLUDE_PATTERNS):
                continue
            models.append(
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    owned_by=model.get("owned_by"),
                    type="llm",
                )
            )

        self._llm_models_cache = models
        logger.info(
            "Fetched LLM models", count=len(models), endpoint=settings.LLM_API_BASE
        )
        return models

    async def get_embedding_models(self, refresh: bool = False) -> list[ModelInfo]:
        """Get available embedding models from the configured endpoint."""
        if self._embedding_models_cache is not None and not refresh:
            return self._embedding_models_cache

        self._embedding_fetch_error = None
        models_data, err = await self._fetch_models(
            settings.EMBEDDING_API_BASE,
            settings.EMBEDDING_API_KEY,
        )
        if err:
            self._embedding_fetch_error = err
            logger.warning(
                "Embedding models fetch failed",
                endpoint=settings.EMBEDDING_API_BASE,
                error=err,
            )
            return []

        models = []
        for model in models_data:
            model_id = model.get("id", "")
            models.append(
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    owned_by=model.get("owned_by"),
                    type="embedding",
                )
            )

        self._embedding_models_cache = models
        logger.info(
            "Fetched embedding models",
            count=len(models),
            endpoint=settings.EMBEDDING_API_BASE,
        )
        return models

    def clear_cache(self):
        """Clear the model cache."""
        self._llm_models_cache = None
        self._embedding_models_cache = None
        self._llm_fetch_error = None
        self._embedding_fetch_error = None

    def get_last_llm_error(self) -> str | None:
        """Return the last LLM models fetch error, if any."""
        return self._llm_fetch_error


# Singleton instance
model_service = ModelService()
