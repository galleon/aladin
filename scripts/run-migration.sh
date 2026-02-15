#!/bin/bash
# Run database migration
# Usage: ./scripts/run-migration.sh [migration_file]

set -e

MIGRATION_FILE=${1:-"migrations/001_add_vlm_and_processing_type.sql"}

if [ ! -f "$MIGRATION_FILE" ]; then
    echo "Error: Migration file not found: $MIGRATION_FILE"
    exit 1
fi

echo "Running migration: $MIGRATION_FILE"

# Check if running in docker or locally
if docker ps | grep -q "aladin-postgres"; then
    echo "Running migration in Docker container..."
    docker exec -i aladin-postgres-1 psql -U postgres -d ragplatform < "$MIGRATION_FILE"
elif docker ps | grep -q "postgres"; then
    # Try to find postgres container
    POSTGRES_CONTAINER=$(docker ps --format "{{.Names}}" | grep -i postgres | head -n 1)
    if [ -n "$POSTGRES_CONTAINER" ]; then
        echo "Running migration in Docker container: $POSTGRES_CONTAINER"
        docker exec -i "$POSTGRES_CONTAINER" psql -U postgres -d ragplatform < "$MIGRATION_FILE"
    else
        echo "Error: Could not find postgres container"
        exit 1
    fi
else
    # Run locally (assuming postgres is accessible)
    echo "Running migration locally..."
    PGHOST=${DB_HOST:-localhost}
    PGPORT=${DB_PORT:-5433}
    PGDATABASE=${DB_NAME:-ragplatform}
    PGUSER=${DB_USER:-postgres}
    PGPASSWORD=${DB_PASSWORD:-postgres}
    
    export PGHOST PGPORT PGDATABASE PGUSER PGPASSWORD
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -f "$MIGRATION_FILE"
fi

echo "Migration completed successfully!"
