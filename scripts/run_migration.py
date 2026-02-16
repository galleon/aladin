#!/usr/bin/env python3
"""Run database migration script."""
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.config import settings
from sqlalchemy import create_engine, text
import structlog

logger = structlog.get_logger()


def run_migration(migration_file: str):
    """Run a SQL migration file."""
    migration_path = Path(__file__).resolve().parent.parent / migration_file
    
    if not migration_path.exists():
        logger.error("Migration file not found", path=str(migration_path))
        return False
    
    logger.info("Running migration", file=str(migration_path))
    
    # Create engine
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
    )
    
    try:
        with engine.connect() as conn:
            # Read migration file
            sql = migration_path.read_text()
            
            # Execute migration
            conn.execute(text(sql))
            conn.commit()
            
        logger.info("Migration completed successfully")
        return True
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        return False
    finally:
        engine.dispose()


if __name__ == "__main__":
    migration_file = sys.argv[1] if len(sys.argv) > 1 else "migrations/001_add_vlm_and_processing_type.sql"
    
    success = run_migration(migration_file)
    sys.exit(0 if success else 1)
