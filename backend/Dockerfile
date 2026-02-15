# RAG Agent Platform Backend Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy pyproject.toml
COPY backend/pyproject.toml ./

# Create virtual environment and install dependencies
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
    "bcrypt>=4.0.1,<4.1.0" \
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
    ffmpeg-python>=0.2.0

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for marker-pdf, document processing, and weasyprint
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

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY backend/app ./app
# Copy prompt library (default path: /app/config/video_prompts.json)
COPY backend/config ./config

# Create uploads directory and subdirectories
RUN mkdir -p /app/uploads/translations /app/uploads/translation_uploads /app/uploads/videos

# Expose port
EXPOSE 3000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
