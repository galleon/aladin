# Canonical worker Dockerfile — used by docker-compose (docker-compose.yml → worker service).
# Builds backend app + pipeline code and runs the ARQ unified worker.
# For the standalone worker image (without backend app), see worker/worker/Dockerfile.
#
# Unified Worker image: ARQ worker (transcription + ingestion).
# Builds backend app + pipeline code and installs worker-only deps (crawl4ai, docling, opencv).
# Backend API uses backend/Dockerfile (no crawl4ai/docling/opencv).

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Backend app deps (API + ARQ entrypoint, transcription)
COPY backend/pyproject.toml ./
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    sqlalchemy>=2.0.23 \
    psycopg2-binary>=2.9.9 \
    "pydantic[email]>=2.5.3" \
    pydantic-settings>=2.1.0 \
    structlog>=24.1.0 \
    python-jose[cryptography]>=3.3.0 \
    "passlib[bcrypt]>=1.7.4" \
    "bcrypt<5.0.0" \
    python-multipart>=0.0.6 \
    qdrant-client>=1.7.0 \
    openai>=1.10.0 \
    anthropic>=0.18.0 \
    cohere>=4.44 \
    langchain>=0.1.0 \
    langchain-openai>=0.0.5 \
    langchain-anthropic>=0.1.0 \
    langchain-community>=0.0.20 \
    langchain-text-splitters>=0.0.1 \
    langgraph>=0.0.20 \
    networkx>=3.0 \
    httpx>=0.26.0 \
    markdown>=3.5.0 \
    "marker-pdf[full]>=1.0.0" \
    redis>=5.0.0 \
    arq>=0.25.0 \
    weasyprint>=60.0 \
    ffmpeg-python>=0.2.0 \
    crawl4ai>=0.4.0 \
    docling>=2.0.0 \
    opencv-python-headless>=4.8.0 \
    ultralytics>=8.0.0 \
    lap>=0.5.12 \
    inference-sdk>=1.0.0 \
    supervision>=0.20.0 \
    opentelemetry-api>=1.22.0 \
    opentelemetry-sdk>=1.22.0 \
    opentelemetry-exporter-otlp-proto-grpc>=1.22.0 \
    opentelemetry-instrumentation-httpx>=0.43b0 \
    opentelemetry-instrumentation-redis>=0.43b0 \
    pypdf>=4.0.0

# Production stage
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    libgl1 \
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    gir1.2-gdkpixbuf-2.0 \
    libffi-dev \
    fonts-liberation \
    fonts-dejavu-core \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Backend app (ARQ unified worker entrypoint lives here)
COPY backend/app ./app
# Pipeline code (web/file/video processors; PYTHONPATH will include this dir)
COPY worker ./worker

RUN mkdir -p /app/uploads/translations /app/uploads/translation_uploads /app/uploads/videos /app/temp

# Worker runs ARQ; pipeline is imported from ingestion when handlers run
CMD ["arq", "app.workers.unified.WorkerSettings"]
