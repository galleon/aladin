"""
Web ingestion: crawl, chunk, embed, Qdrant storage.
"""
from .processor import WebProcessor
from .crawler import crawl_pages, extract_tables, table_to_markdown

__all__ = [
    "WebProcessor",
    "crawl_pages",
    "extract_tables",
    "table_to_markdown",
]
