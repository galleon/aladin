# Aladin

A full-stack RAG (Retrieval-Augmented Generation) platform for building AI assistants powered by your own documents. Whether your data lives in PDFs, web pages, audio, or video, Aladin lets you ingest, search, and chat with it through a single interface.

## Features

- **Document RAG** — Upload PDFs, DOCX, TXT, MD, CSV, JSON, or crawl web pages. Documents are chunked, embedded, and stored in Qdrant for semantic search. Chat with agents that cite sources with relevance scores.
- **Data Quality Validation** — Comprehensive validation for all uploaded files including MIME type verification, UTF-8 encoding checks, file size limits, duplicate detection with SHA-256 checksums, and file corruption detection.
- **Video transcription** — Transcribe video/audio files via Whisper API. Optionally analyse segments with a vision LLM (VLM) and YOLO object tracking.
- **Document translation** — Translate documents between languages using any OpenAI-compatible LLM.
- **Multi-LLM support** — Any OpenAI-compatible endpoint (OpenAI, Anthropic via proxy, LiteLLM, local models). Models are fetched dynamically from the configured endpoint.
- **Async job processing** — Heavy work (ingestion, transcription, video analysis) runs on ARQ workers backed by Redis. Scale horizontally with `docker-compose up --scale worker=N`.
- **Auth** — JWT-based authentication with OAuth2 password flow.

## Architecture

```
Frontend (React/TS, port 5174)
  │
  ├── Backend (FastAPI, port 3000)
  |       |── PostgreSQL (metadata)
  │       │── Qdrant (vector search)
  │       │── Redis (job queue)
  │       │── LLM APIs
  │
  └── Worker (ARQ)
          |── document ingestion
          |── video transcription + VLM analysis
          |── web crawling
```

Additional services: chat-ui (port 7860), Jaeger tracing (optional, `observability` profile).

## Quick start

### Prerequisites

- Docker and Docker Compose
- An OpenAI-compatible API endpoint (OpenAI, LiteLLM, local server, etc.)

### Setup

```bash
cp env.local.template .env
# Edit .env — at minimum set SECRET_KEY and LLM/embedding endpoints

docker-compose up --build
```

The application will be available at:
- **Frontend**: http://localhost:5174
- **Backend API**: http://localhost:3000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Usage

1. Register an account and log in
2. Create a **Data Domain** — upload documents or crawl a URL
3. Create an **Agent** — pick a model, write a system prompt, link data domains
4. **Chat** — ask questions and get answers with source citations

## Project structure

```
backend/              FastAPI application
  app/
    main.py           App init, router mounting
    models.py         SQLAlchemy ORM models
    schemas.py        Pydantic request/response schemas
    config.py         Settings from environment variables
    routers/          REST endpoints (auth, agents, conversations, data_domains,
                      ingestion, jobs, models, stats, translation, video_transcription)
    services/         Business logic (RAG, embeddings, Qdrant, auth, document
                      processing, translation, video transcription, file_validation)

frontend/             React SPA (Vite + TypeScript + Tailwind)
  src/
    App.tsx           Router with protected routes
    api/client.ts     Axios client with JWT interceptor
    pages/            Dashboard, Agents, Chat, DataDomains, Conversations, etc.
    components/       Layout, ErrorModal, VideoTranscription

worker/               ARQ job handlers (transcription, ingestion, video analysis)
                     CLI: python -m worker file|video|web [args...]

chat-ui/              Standalone chat interface (port 7860)
migrations/           Numbered SQL migration files
scripts/              Utility scripts (migration runners, stack restart)
```

## Tech stack

| Layer | Technologies |
|-------|-------------|
| Backend | FastAPI, SQLAlchemy 2.0, ARQ, LangChain/LangGraph, structlog |
| Frontend | React 18, TypeScript, Tailwind CSS, Vite, React Query, React Router 6 |
| Worker | ARQ, docling, marker-pdf, crawl4ai, ffmpeg, opencv, ultralytics (YOLO) |
| Infrastructure | PostgreSQL, Qdrant, Redis, Docker Compose |
| Python packaging | uv |

## Environment variables

See `env.local.template` for the full list. Key variables:

| Variable | Purpose |
|----------|---------|
| `SECRET_KEY` | JWT signing key (min 32 chars) |
| `LLM_API_BASE` / `LLM_API_KEY` | OpenAI-compatible LLM endpoint |
| `EMBEDDING_API_BASE` / `EMBEDDING_MODEL` | Embedding endpoint and model name |
| `WHISPER_API_BASE` | Whisper API for video transcription (optional) |
| `VLM_API_BASE` / `VLM_MODEL` | Vision LLM for video analysis (optional) |
| `DB_*`, `QDRANT_*`, `REDIS_*` | Service connection settings |
| `MAX_FILE_SIZE` | Maximum file size in bytes (default: 52428800 = 50MB) |
| `MAX_VIDEO_SIZE` | Maximum video file size in bytes (default: 524288000 = 500MB) |

## Data Quality Validation

Aladin includes comprehensive data quality validation for all uploaded files to ensure data integrity and prevent security issues:

### Validation Checks

1. **File Type Validation**
   - Verifies file extensions against allowed types
   - Validates MIME types using magic bytes (file signatures)
   - Supported formats: PDF, DOCX, PPTX, DOC, TXT, MD, HTML, CSV, JSON, MP4

2. **Data Encoding**
   - Text-based files (TXT, MD, HTML, CSV, JSON) must be UTF-8 encoded
   - Prevents data corruption from invalid encodings
   - Binary files (PDF, DOCX, MP4) are not subject to encoding checks

3. **File Size Limits**
   - Minimum size: 10 bytes (prevents empty files)
   - Maximum size for documents: 50MB (configurable via `MAX_FILE_SIZE`)
   - Maximum size for videos: 500MB (configurable via `MAX_VIDEO_SIZE`)

4. **Duplicate Detection**
   - SHA-256 checksums calculated for all uploaded files
   - Prevents duplicate file uploads within the same session
   - Checksum included in file metadata for integrity verification

5. **File Corruption Detection**
   - Magic byte verification ensures file headers match expected formats
   - Detects renamed or corrupted files (e.g., .txt file with PDF extension)

### Error Responses

When validation fails, the API returns a 400 Bad Request with a descriptive error message:

- `EMPTY_FILE`: File contains no data
- `FILE_TOO_SMALL`: File is smaller than minimum size (10 bytes)
- `FILE_TOO_LARGE`: File exceeds maximum size limit
- `INVALID_FILE_TYPE`: File extension not supported or missing
- `INVALID_MIME_TYPE`: File content doesn't match extension (e.g., PDF data with .txt extension)
- `INVALID_ENCODING`: Text file is not valid UTF-8
- `DUPLICATE_FILE`: File with identical content already uploaded

### Configuration

Configure file size limits in your `.env` file:

```bash
# Maximum file size for documents (bytes)
MAX_FILE_SIZE=52428800  # 50MB

# Maximum file size for videos (bytes)  
MAX_VIDEO_SIZE=524288000  # 500MB
```

## License

MIT
