"""Qdrant vector store service."""
import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import structlog

from ..config import settings

logger = structlog.get_logger()

# Embedding dimensions for different models
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embeddings": 1024,  # User's custom embedding model
}


class QdrantService:
    """Service for managing Qdrant vector store operations."""

    def __init__(self):
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
            )
        return self._client

    def create_collection(self, collection_name: str, embedding_model: str) -> bool:
        """Create a new Qdrant collection for a data domain."""
        try:
            # Try to get dimension from cache/dictionary first
            vector_size = EMBEDDING_DIMENSIONS.get(embedding_model)

            # If not found, dynamically detect the dimension by testing the embedding model
            if vector_size is None:
                logger.info(
                    "Embedding model dimension not cached, detecting dynamically",
                    model=embedding_model
                )
                try:
                    from ..services.embedding_service import embedding_service
                    # Generate a test embedding to get the actual dimension
                    test_embedding = embedding_service.embed_texts(["test"], embedding_model)
                    if test_embedding and len(test_embedding) > 0:
                        vector_size = len(test_embedding[0])
                        # Cache it for future use
                        EMBEDDING_DIMENSIONS[embedding_model] = vector_size
                        logger.info(
                            "Detected embedding dimension",
                            model=embedding_model,
                            dimension=vector_size
                        )
                    else:
                        raise ValueError("Could not determine embedding dimension")
                except Exception as e:
                    logger.warning(
                        "Failed to detect embedding dimension, using default",
                        model=embedding_model,
                        error=str(e),
                        default=1536
                    )
                    vector_size = 1536  # Fallback to default

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection", collection=collection_name, vector_size=vector_size, model=embedding_model)
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warn("Collection already exists", collection=collection_name)
                return True
            logger.error("Failed to create collection", collection=collection_name, error=str(e))
            raise

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info("Deleted Qdrant collection", collection=collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection", collection=collection_name, error=str(e))
            return False

    def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None
    ) -> bool:
        """Upsert vectors into a collection."""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            points = [
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload
                )
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            logger.info("Upserted vectors", collection=collection_name, count=len(vectors))
            return True
        except Exception as e:
            logger.error("Failed to upsert vectors", collection=collection_name, error=str(e))
            raise

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: dict | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        try:
            # Build filter if needed
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )

            # Use the correct Qdrant client query_points method
            # query_points accepts the vector directly and other parameters
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Extract points from the QueryResponse
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in response.points
            ]
        except Exception as e:
            logger.error("Search failed", collection=collection_name, error=str(e))
            raise

    def delete_by_document(self, collection_name: str, document_id: int) -> bool:
        """Delete all vectors for a specific document (payload has document_id)."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            logger.info("Deleted vectors for document", collection=collection_name, document_id=document_id)
            return True
        except Exception as e:
            logger.error("Failed to delete vectors", collection=collection_name, document_id=document_id, error=str(e))
            return False

    def delete_by_source_file(self, collection_name: str, source_file: str) -> bool:
        """Delete all vectors whose payload.source_file matches (e.g. for reindex)."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=source_file),
                            )
                        ]
                    )
                ),
            )
            logger.info("Deleted vectors by source_file", collection=collection_name, source_file=source_file)
            return True
        except Exception as e:
            logger.error(
                "Failed to delete vectors by source_file",
                collection=collection_name,
                source_file=source_file,
                error=str(e),
            )
            return False

    def get_collection_info(self, collection_name: str) -> dict[str, Any] | None:
        """Get collection information."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error("Failed to get collection info", collection=collection_name, error=str(e))
            return None

    def scroll(
        self,
        collection_name: str,
        limit: int = 20,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Scroll through points in a collection. Returns (items, next_offset)."""
        try:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            items = [
                {
                    "id": str(p.id),
                    "payload": p.payload or {},
                    **({"vector": p.vector} if with_vectors and p.vector else {}),
                }
                for p in points
            ]
            return (items, next_offset)
        except Exception as e:
            logger.error("Scroll failed", collection=collection_name, error=str(e))
            raise


# Singleton instance
qdrant_service = QdrantService()
