"""
Shared schemas for ingestion pipeline.
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime


class CrawlStrategy(str, Enum):
    """Crawl strategy for web ingestion."""
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search


class TableStrategy(str, Enum):
    """Strategy for table extraction."""
    RAW_HTML = "raw_html"  # Keep raw HTML
    MARKDOWN = "markdown"  # Convert to markdown
    SUMMARY_AND_HTML = "summary_and_html"  # Summarize + keep HTML


class WebSourceConfig(BaseModel):
    """Configuration for web source crawling."""
    url: HttpUrl = Field(..., description="Starting URL to crawl")
    depth_limit: int = Field(default=2, ge=0, le=10, description="Maximum crawl depth")
    strategy: CrawlStrategy = Field(default=CrawlStrategy.BFS, description="Crawl strategy")
    inclusion_patterns: List[str] = Field(default_factory=list, description="URL patterns to include")
    exclusion_patterns: List[str] = Field(default_factory=list, description="URL patterns to exclude")
    max_pages: int = Field(default=100, ge=1, le=1000, description="Maximum pages to crawl")


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    render_js: bool = Field(default=False, description="Render JavaScript before extracting")
    wait_for_selector: Optional[str] = Field(default=None, description="CSS selector to wait for")
    wait_timeout: int = Field(default=30, ge=5, le=120, description="Wait timeout in seconds")
    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    table_strategy: TableStrategy = Field(default=TableStrategy.MARKDOWN, description="Table extraction strategy")
    extract_images: bool = Field(default=False, description="Extract and describe images")
    vlm_model: Optional[str] = Field(default=None, description="VLM model for image description")
    extract_links: bool = Field(default=True, description="Extract links from documents")
    clean_html: bool = Field(default=True, description="Clean and normalize HTML")


class WebIngestionRequest(BaseModel):
    """Request to ingest web content."""
    job_id: Optional[str] = Field(default=None, description="Custom job ID")
    source: WebSourceConfig
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    collection_name: str = Field(..., min_length=1, max_length=100, description="Target Qdrant collection")
    metadata: dict = Field(default_factory=dict, description="Additional metadata to attach")


class FileIngestionRequest(BaseModel):
    """Request to ingest a file."""
    job_id: Optional[str] = Field(default=None, description="Custom job ID")
    collection_name: str = Field(..., min_length=1, max_length=100, description="Target Qdrant collection")
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    metadata: dict = Field(default_factory=dict, description="Additional metadata to attach")


class IngestionJobStatus(str, Enum):
    """Status of an ingestion job."""
    QUEUED = "queued"
    CRAWLING = "crawling"
    EXTRACTING = "extracting"
    PARTITIONING = "partitioning"
    SUMMARIZING = "summarizing"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJobResponse(BaseModel):
    """Response for an ingestion job."""
    job_id: str
    status: IngestionJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    pages_processed: int = 0
    pages_total: int = 0
    chunks_created: int = 0
    error_message: Optional[str] = None
    trace_id: Optional[str] = None


class DocumentChunk(BaseModel):
    """A chunk of a document ready for embedding."""
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    source_url: Optional[str] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 1


class IngestionMetrics(BaseModel):
    """Metrics for an ingestion job."""
    job_id: str
    crawl_duration_ms: Optional[float] = None
    partition_duration_ms: Optional[float] = None
    summarization_duration_ms: Optional[float] = None
    embedding_duration_ms: Optional[float] = None
    total_duration_ms: Optional[float] = None
    pages_crawled: int = 0
    tables_extracted: int = 0
    images_processed: int = 0
    chunks_created: int = 0
    extraction_failures: int = 0




