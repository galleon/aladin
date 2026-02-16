"""Embedding service for generating vector embeddings."""

from __future__ import annotations

import structlog
from openai import OpenAI

from ..config import settings

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating embeddings from text using OpenAI-compatible API."""

    def __init__(self):
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI-compatible client for embeddings."""
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_API_BASE,
            )
        return self._client

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=model,
            )

            # Check if we got valid embedding data
            if not response.data or len(response.data) == 0:
                error_msg = f"No embedding data received from API. Model '{model}' may not support embeddings or the API returned an empty response."
                logger.error(
                    "Embedding failed - no data",
                    model=model,
                    base_url=settings.EMBEDDING_API_BASE,
                    response_type=type(response).__name__,
                )
                raise ValueError(error_msg)

            embeddings = [item.embedding for item in response.data]

            # Validate embeddings are not empty
            if not embeddings or any(not emb or len(emb) == 0 for emb in embeddings):
                error_msg = f"Received empty embeddings from model '{model}'. This model may not support embeddings."
                logger.error(
                    "Embedding failed - empty embeddings",
                    model=model,
                    base_url=settings.EMBEDDING_API_BASE,
                )
                raise ValueError(error_msg)

            logger.info(
                "Generated embeddings",
                model=model,
                count=len(texts),
                base_url=settings.EMBEDDING_API_BASE,
            )
            return embeddings
        except ValueError:
            # Re-raise ValueError as-is (our custom errors)
            raise
        except Exception as e:
            # For other exceptions, provide more context
            error_msg = f"Embedding API call failed: {str(e)}. Model: '{model}', API: {settings.EMBEDDING_API_BASE}"
            logger.error(
                "Embedding failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
                base_url=settings.EMBEDDING_API_BASE,
            )
            raise ValueError(error_msg) from e

    def embed_query(self, query: str, model: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.embed_texts([query], model)[0]


# Singleton instance
embedding_service = EmbeddingService()
