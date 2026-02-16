"""Configuration settings for RAG Agent Management Platform."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "RAG Agent Platform"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    CORS_ORIGINS: list[str] = (
        os.getenv("CORS_ORIGINS", "*").split(",")
        if os.getenv("CORS_ORIGINS")
        else ["*"]
    )

    # Database
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "ragplatform")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # Authentication
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "your-secret-key-change-in-production-min-32-chars"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
    )  # 24 hours

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str | None = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")

    # LLM API (OpenAI-compatible endpoint)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "sk-dummy-key")

    # Embedding API (OpenAI-compatible endpoint, can be same or different from LLM)
    EMBEDDING_API_BASE: str = os.getenv(
        "EMBEDDING_API_BASE", "http://localhost:8000/v1"
    )
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "sk-dummy-key")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # Whisper API (required - external transcription service endpoint)
    # Video transcription agents require this to be configured
    WHISPER_API_BASE: str | None = os.getenv("WHISPER_API_BASE", None)
    WHISPER_API_KEY: str | None = os.getenv("WHISPER_API_KEY", None)

    # OpenTelemetry
    OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )

    # File uploads
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB
    MAX_VIDEO_SIZE: int = int(os.getenv("MAX_VIDEO_SIZE", "524288000"))  # 500MB
    ALLOWED_EXTENSIONS: set = {"pdf", "txt", "md", "docx", "doc", "csv", "json", "mp4"}

    # VLM API (for video processing - optional, can be per-domain)
    VLM_API_BASE: str | None = os.getenv("VLM_API_BASE", None)
    VLM_API_KEY: str | None = os.getenv("VLM_API_KEY", None)
    VLM_MODEL: str | None = os.getenv("VLM_MODEL", None)

    # Prompt library: path to JSON file with keyword -> template (e.g. procedure, race)
    VIDEO_PROMPT_LIBRARY_PATH: str | None = os.getenv("VIDEO_PROMPT_LIBRARY_PATH", None)

    # Marker PDF settings
    MARKER_USE_LLM: bool = os.getenv("MARKER_USE_LLM", "false").lower() == "true"

    # RAG settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Email (for transcription job completion notification)
    EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    SMTP_HOST: str | None = os.getenv("SMTP_HOST")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str | None = os.getenv("SMTP_USER")
    SMTP_PASSWORD: str | None = os.getenv("SMTP_PASSWORD")
    SMTP_FROM: str | None = os.getenv("SMTP_FROM")
    FRONTEND_BASE_URL: str = os.getenv("FRONTEND_BASE_URL", "http://localhost:5174")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

# Default SECRET_KEY value for validation
DEFAULT_SECRET_KEY = "your-secret-key-change-in-production-min-32-chars"

# Fallback prompt templates when VIDEO_PROMPT_LIBRARY_PATH is missing or invalid
_FALLBACK_PROMPT_LIBRARY = {
    "procedure": "Analyze this procedure video segment [{t_start}s–{t_end}s] using {num_frames} keyframes. "
    "For each step, provide:\n- **action**: what is being done\n- **tools**: tools/equipment used\n"
    "- **result**: outcome or visible state change\n- **warnings**: safety or quality cues if any\n"
    "- **visible_cues**: text, labels, or indicators on screen\n"
    'Respond with a structured JSON: { "caption": "...", "events": [...], "entities": [...], "notes": [...] }.',
    "race": "Analyze this race/traffic video segment [{t_start}s–{t_end}s] using {num_frames} keyframes. Provide:\n"
    "- **caption**: scene-level summary\n"
    "- **events**: interactions (e.g. closing gap, overtake attempt, lane change), "
    "with per-track references when possible (e.g. 'Car T07', 'Pedestrian P01')\n"
    "- **entities**: vehicles, pedestrians, or other moving objects with stable IDs if known\n"
    "- **notes**: uncertainty when objects are occluded or leave frame\n"
    'Respond with a structured JSON: { "caption": "...", "events": [...], "entities": [...], "notes": [...] }.',
}


def get_video_prompt_library() -> dict[str, str]:
    """Load prompt library from JSON file. Returns keyword -> template. Falls back to built-in if missing/invalid."""
    path = settings.VIDEO_PROMPT_LIBRARY_PATH
    if not path:
        # Default: backend/config/video_prompts.json relative to this file (backend/app/config.py)
        path = str(Path(__file__).resolve().parent.parent / "config" / "video_prompts.json")
    path = Path(path)
    if not path.exists():
        return _FALLBACK_PROMPT_LIBRARY.copy()
    try:
        with open(path, encoding="utf-8") as f:
            data: Any = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
            return dict(data)
    except (json.JSONDecodeError, OSError):
        pass
    return _FALLBACK_PROMPT_LIBRARY.copy()
