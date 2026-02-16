"""
File ingestion: document extraction, chunking, embedding, Qdrant storage.
"""
from .processor import FileProcessor
from .loader import (
    load_text_file,
    load_json_file,
    call_external_docling,
    get_pdf_metadata,
    fallback_pdf_extraction,
)
from .chunker import create_chunks_from_content

__all__ = [
    "FileProcessor",
    "load_text_file",
    "load_json_file",
    "call_external_docling",
    "get_pdf_metadata",
    "fallback_pdf_extraction",
    "create_chunks_from_content",
]
