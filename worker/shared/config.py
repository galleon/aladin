"""
Shared configuration for ingestion services.

NOTE: This config is intentionally separate from backend/app/config.py.
The worker runs in a different container/runtime and has worker-specific
settings (e.g. DOCLING_*, YOLO_*, WORKER_CONCURRENCY) that the API server
does not need. Overlapping settings (DB, Redis, Qdrant, LLM) are read from
the same environment variables so they stay in sync via docker-compose.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Ingestion pipeline settings."""

    # Service identification
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "ingestion")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Database configuration (PostgreSQL)
    DB_HOST: str = os.getenv("DB_HOST", "postgres")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "ragplatform")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # Qdrant configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # LLM/Embedding API configuration
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "sk-dummy-key")
    EMBEDDING_API_BASE: str = os.getenv(
        "EMBEDDING_API_BASE", "http://localhost:8000/v1"
    )
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "sk-dummy-key")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    # Max INPUT length (chars) per text sent to embedding API. Many models (e.g. bge) have max 512 INPUT tokens;
    # 1024 chars is safe for 512-token models; increase only if your embedding model supports longer context.
    EMBEDDING_MAX_INPUT_CHARS: int = int(os.getenv("EMBEDDING_MAX_INPUT_CHARS", "1024"))

    # Docling: external REST API (optional). When set, file ingestion uses this instead of in-process docling.
    # If the base URL is OpenAI-compatible (e.g. /v1/models lists granite_docling), we call /v1/chat/completions with DOCLING_MODEL.
    DOCLING_API_BASE: Optional[str] = os.getenv("DOCLING_API_BASE") or os.getenv(
        "LLM_API_BASE"
    )
    # Docling model name (external API e.g. granite_docling, or in-process HybridChunker e.g. ibm-granite/granite-docling-258M)
    DOCLING_MODEL: str = os.getenv("DOCLING_MODEL", "granite_docling")

    # VLM configuration for image extraction
    VLM_API_BASE: str = os.getenv("VLM_API_BASE", "http://localhost:8000/v1")
    VLM_API_KEY: str = os.getenv("VLM_API_KEY", "sk-dummy-key")
    VLM_MODEL: str = os.getenv("VLM_MODEL", "llava-v1.6-34b")

    # YOLO/Roboflow Inference API for object detection (video tracking)
    YOLO_API_URL: Optional[str] = os.getenv("YOLO_API_URL")
    YOLO_API_KEY: str = os.getenv("YOLO_API_KEY", "")
    YOLO_MODEL_ID: str = os.getenv("YOLO_MODEL_ID", "yolov8n-640/1")

    # Storage paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/data/uploads")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/data/temp")

    # OpenTelemetry configuration
    OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"
    )
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "ingestion-pipeline")

    # Worker configuration
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "5"))
    JOB_TIMEOUT: int = int(os.getenv("JOB_TIMEOUT", "600"))  # 10 minutes

    # VLM outcome logging for review (video ingestion)
    LOG_VLM_OUTCOME: bool = os.getenv("LOG_VLM_OUTCOME", "false").lower() == "true"
    # Log every Nth segment (1 = all, 5 = every 5th) to limit volume
    LOG_VLM_OUTCOME_SAMPLE_EVERY: int = int(os.getenv("LOG_VLM_OUTCOME_SAMPLE_EVERY", "1"))
    # When set, write vlm_out to sidecar JSONL file for file-based review
    LOG_VLM_OUTCOME_REVIEW_FILE: bool = os.getenv("LOG_VLM_OUTCOME_REVIEW_FILE", "false").lower() == "true"

    # Crawling defaults
    DEFAULT_CRAWL_DEPTH: int = int(os.getenv("DEFAULT_CRAWL_DEPTH", "2"))
    DEFAULT_CRAWL_STRATEGY: str = os.getenv("DEFAULT_CRAWL_STRATEGY", "bfs")

    # Text splitting configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def truncate_for_embedding(text: str) -> str:
    """Truncate input text to fit model's max INPUT tokens (e.g. bge-large 512). Output embedding dimension is unchanged."""
    if not text:
        return text
    max_chars = getattr(settings, "EMBEDDING_MAX_INPUT_CHARS", 1024)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(maxsplit=1)[0] or text[:max_chars]
