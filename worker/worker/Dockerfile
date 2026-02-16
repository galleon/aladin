# Standalone ingestion worker Dockerfile — for running the worker independently of docker-compose.
# The canonical docker-compose worker uses worker.Dockerfile at the repository root instead.
#
# Ingestion Worker Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies for crawl4ai, docling, and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    curl \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy shared config and worker code
COPY shared /app/shared
COPY worker /app/worker

# Create required directories
RUN mkdir -p /app/uploads /app/temp

# Install Python dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache \
    "pydantic>=2.0.0" \
    "pydantic-settings>=2.0.0" \
    "redis>=5.0.0" \
    "arq>=0.25.0" \
    "crawl4ai>=0.3.0" \
    "beautifulsoup4>=4.12.0" \
    "docling>=2.0.0" \
    "langchain-docling>=0.1.0" \
    "langchain-text-splitters>=0.0.1" \
    "qdrant-client>=1.7.0" \
    "openai>=1.0.0" \
    "pypdf>=4.0.0" \
    "opentelemetry-api>=1.22.0" \
    "opentelemetry-sdk>=1.22.0" \
    "opentelemetry-exporter-otlp>=1.22.0" \
    "opentelemetry-instrumentation-httpx>=0.43b0" \
    "opentelemetry-instrumentation-redis>=0.43b0" \
    "sqlalchemy>=2.0.23" \
    "psycopg2-binary>=2.9.9" \
    "numpy>=1.24.0" \
    "opencv-python-headless>=4.8.0"

# Note: Playwright will be installed on first crawl4ai use
# Skip pre-installing to save space in Docker image

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Run the worker
CMD ["python", "-m", "arq", "worker.main.WorkerSettings"]
