"""
File ingestion: document extraction, chunking, embedding, Qdrant storage.

Production path: RichFileProcessor (nv-ingest-style — bboxes, tables, image captions).
FileProcessor is kept only for the CLI dev tool (extract_and_chunk without embedding).
"""
from .rich_processor import RichFileProcessor
from .loader import (
    load_text_file,
    load_json_file,
    call_external_docling,
    get_pdf_metadata,
    fallback_pdf_extraction,
)
from .chunker import create_chunks_from_content

__all__ = [
    "RichFileProcessor",
    "load_text_file",
    "load_json_file",
    "call_external_docling",
    "get_pdf_metadata",
    "fallback_pdf_extraction",
    "create_chunks_from_content",
]
