"""Database configuration and session management for ingestion worker."""
import sys
import os
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Database URL from settings
DATABASE_URL = settings.DATABASE_URL

# Create engine with connection timeout
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    pool_pre_ping=True,
    echo=False,
    connect_args={"connect_timeout": 10},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session():
    """
    Get a database session with automatic cleanup.
    Usage:
        with get_db_session() as session:
            # use session
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_backend_models():
    """
    Import backend models by adding backend to path.
    Returns (Document, DataDomain). Works when run from backend container (/app) or ingestion worker.
    """
    # When run from unified worker (backend container), app is already at /app
    if "/app" in sys.path or str(Path("/app")) in sys.path:
        try:
            from app.models import Document, DataDomain
            return Document, DataDomain
        except ImportError:
            pass

    # Try mounted backend path (ingestion worker Docker: ./backend/app -> /app/backend_app)
    mounted_backend = Path("/app/backend_app")
    if mounted_backend.exists() and (mounted_backend / "models.py").exists():
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        try:
            from app.models import Document, DataDomain
            return Document, DataDomain
        except ImportError:
            pass

    # Relative path (local development)
    backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    try:
        from app.models import Document, DataDomain
        return Document, DataDomain
    except ImportError as e:
        logger.error(f"Failed to import backend models: {e}")
        raise
