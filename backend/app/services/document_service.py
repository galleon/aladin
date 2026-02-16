"""Document file and vector operations (ingestion runs in the worker)."""

import os
import uuid
from pathlib import Path
import structlog

from ..config import settings
from .qdrant_service import qdrant_service

logger = structlog.get_logger()


class DocumentService:
    """Service for saving/deleting document files and deleting vectors."""

    def delete_document_vectors(self, document_id: int, collection_name: str) -> bool:
        """Delete all vectors for a document from the collection (by document_id in payload)."""
        return qdrant_service.delete_by_document(collection_name, document_id)

    def delete_vectors_by_source_file(self, collection_name: str, source_file: str) -> bool:
        """Delete all vectors whose payload.source_file matches (for reindex)."""
        return qdrant_service.delete_by_source_file(collection_name, source_file)

    def save_uploaded_file(self, file_content: bytes, original_filename: str) -> str:
        """Save an uploaded file and return the path."""
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_ext = Path(original_filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(
            "Saved uploaded file", original=original_filename, saved_as=unique_filename
        )
        return file_path

    def delete_file(self, file_path: str) -> bool:
        """Delete a file from disk."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Deleted file", path=file_path)
            return True
        except Exception as e:
            logger.error("Failed to delete file", path=file_path, error=str(e))
            return False


document_service = DocumentService()
