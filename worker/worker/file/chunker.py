"""
Chunking: split content into DocumentChunks for embedding.
"""
from __future__ import annotations

import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import settings
from shared.schemas import DocumentChunk


def create_chunks_from_content(
    content: str,
    filename: str,
    file_ext: str,
    job_id: str,
    text_splitter: RecursiveCharacterTextSplitter | None = None,
    doc_meta: dict | None = None,
) -> List[DocumentChunk]:
    """Create document chunks from content string."""
    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    meta = doc_meta or {}
    base_meta = {"source_type": "file", "file_type": file_ext.lstrip("."), **meta}

    chunks: List[DocumentChunk] = []
    texts = text_splitter.split_text(content)

    for i, text in enumerate(texts):
        chunk = DocumentChunk(
            id=f"{job_id}_{uuid.uuid4().hex[:8]}",
            content=text,
            metadata={**base_meta},
            source_file=filename,
            chunk_index=i,
            total_chunks=len(texts),
        )
        chunks.append(chunk)

    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks
