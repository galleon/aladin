"""RAG Agent Management Platform - Main Application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .config import settings
from .database import engine, Base
from .routers import (
    auth,
    data_domains,
    agents,
    conversations,
    models,
    translation,
    ingestion,
    video_transcription,
    jobs,
    stats,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting ALADIN Platform")
    logger.info(
        "Video transcription",
        available=bool(settings.WHISPER_API_BASE),
    )

    # Skip automatic table creation - tables are managed via migrations
    # New tables (chat_sessions, rag_citations) with UUIDs require migration
    # and will cause foreign key type mismatches if auto-created
    logger.info("Database tables should be migrated manually - skipping auto-creation")

    yield

    logger.info("Shutting down ALADIN Platform")


app = FastAPI(
    title=settings.APP_NAME,
    description="A platform for managing RAG agents and data domains",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(data_domains.router, prefix="/api")
app.include_router(agents.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(translation.router, prefix="/api")
app.include_router(ingestion.router, prefix="/api")
app.include_router(video_transcription.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(stats.router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"name": settings.APP_NAME, "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
