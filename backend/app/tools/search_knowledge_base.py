"""search_knowledge_base tool – wraps Qdrant vector search for RAG retrieval."""

from __future__ import annotations

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger()


@tool
def search_knowledge_base(
    query: str,
    collection_name: str = "",
    embedding_model: str = "",
    k: int = 5,
) -> str:
    """Search the knowledge base for documents relevant to the query.

    Args:
        query: The search query.
        collection_name: Qdrant collection to search in.
        embedding_model: Embedding model used by the collection.
        k: Number of top results to return (default 5).

    Returns:
        Formatted context string with source references.
    """
    from ..services.embedding_service import embedding_service
    from ..services.qdrant_service import qdrant_service

    if not collection_name or not embedding_model:
        return "Error: collection_name and embedding_model are required."

    try:
        query_embedding = embedding_service.embed_query(query, embedding_model)
        results = qdrant_service.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            score_threshold=0.3,
        )

        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(results, 1):
            payload = doc.get("payload", {})
            text = payload.get("content", "") or payload.get("text", "")
            filename = (
                payload.get("source_file", "") or payload.get("filename", "Unknown")
            )
            page = payload.get("page", "N/A")
            context_parts.append(f"[Source {i}: {filename}, Page {page}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "search_knowledge_base tool returned results",
            num_sources=len(results),
            total_chars=len(context),
        )
        return context

    except Exception as e:
        logger.error("search_knowledge_base tool failed", error=str(e))
        return f"Search failed: {e}"
