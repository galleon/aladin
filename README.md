# Aladin

A full-stack RAG (Retrieval-Augmented Generation) platform for building AI assistants powered by your own documents. Whether your data lives in PDFs, web pages, audio, or video, Aladin lets you ingest, search, and chat with it through a single interface.

## Features

- **Document RAG** — Upload PDFs, DOCX, TXT, MD, CSV, JSON, or crawl web pages. Documents are chunked, embedded, and stored in Qdrant for semantic search. Chat with agents that cite sources with relevance scores.
- **Voice chat with VAD** — Talk to your agents using voice input with automatic Voice Activity Detection (VAD). Responses can be read aloud with text-to-speech. Uses OpenAI-compatible STT/TTS APIs.
- **Video transcription** — Transcribe video/audio files via Whisper API. Optionally analyse segments with a vision LLM (VLM) and YOLO object tracking.
- **Document translation** — Translate documents between languages using any OpenAI-compatible LLM.
- **Multi-LLM support** — Any OpenAI-compatible endpoint (OpenAI, Anthropic via proxy, LiteLLM, local models). Models are fetched dynamically from the configured endpoint.
- **Dynamic tool-calling agents** — Agents can be equipped with tools (web ingestion, knowledge-base search, translation) via a ReAct loop powered by LangGraph. Jobs triggered by tools are tracked in PostgreSQL and visible in the job management UI.
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
  │       │── LiteLLM (port 4000) ── spark-1 (granite_docling, cosmos)
  │                                └── spark-2 (thinking, embeddings, reranker)
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
- OpenAI-compatible model endpoints (local vLLM, LiteLLM proxy, OpenAI, etc.)

### Setup

```bash
cp env.local.template .env
# Edit .env — set SECRET_KEY and the two node IPs if using a multi-node GPU setup:
#   SPARK_1_IP=<ip of node 1>
#   SPARK_2_IP=<ip of node 2>

docker-compose up --build
```

LiteLLM starts automatically as part of the stack (port 4000) and routes requests
to the configured model backends via `litellm.yaml`. Backend and worker containers
connect to it using the internal Docker hostname `litellm:4000` — no `.local` mDNS
resolution required.

The application will be available at:
- **Frontend**: http://localhost:5174
- **Backend API**: http://localhost:3000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Usage

1. Register an account and log in
2. Create a **Data Domain** — upload documents or crawl a URL
3. Create an **Agent** — pick a model, write a system prompt, link data domains
4. **Chat** — ask questions and get answers with source citations
5. **Voice Chat** (optional) — Click the microphone button to speak your question. The system will automatically detect when you stop speaking (VAD). Assistant responses can be played aloud using the speaker icon next to each message.

### Voice Chat Configuration

To enable voice chat, set up OpenAI-compatible STT/TTS endpoints:

```bash
# Speech-to-Text (uses Whisper endpoint by default)
STT_API_BASE=https://api.openai.com/v1
STT_API_KEY=sk-your-openai-api-key
STT_MODEL=whisper-1

# Text-to-Speech
TTS_API_BASE=https://api.openai.com/v1
TTS_API_KEY=sk-your-openai-api-key
TTS_MODEL=tts-1
TTS_VOICE=alloy  # Options: alloy, echo, fable, onyx, nova, shimmer
```

If not configured, the voice buttons will still appear but API calls will fail gracefully.

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
                      processing, translation, video transcription, knowledge graph)
    tools/            LangChain @tool definitions for tool-calling agents
                      (search_knowledge_base, translate_text, ingest_url, MCP client)

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
| Infrastructure | PostgreSQL, Qdrant, Redis, LiteLLM, Docker Compose |
| Python packaging | uv |

## Environment variables

See `env.local.template` for the full list. Key variables:

| Variable | Purpose |
|----------|---------|
| `SECRET_KEY` | JWT signing key (min 32 chars) |
| `SPARK_1_IP` | IP of GPU node 1 (injected as `spark-1.internal` in containers) |
| `SPARK_2_IP` | IP of GPU node 2 (injected as `spark-2.internal` in containers) |
| `LLM_API_BASE` / `LLM_API_KEY` | OpenAI-compatible LLM endpoint (overridden to `litellm:4000` in Docker) |
| `EMBEDDING_API_BASE` / `EMBEDDING_MODEL` | Embedding endpoint and model name |
| `STT_API_BASE` / `STT_API_KEY` / `STT_MODEL` | Speech-to-Text API for voice chat |
| `TTS_API_BASE` / `TTS_API_KEY` / `TTS_MODEL` | Text-to-Speech API for voice chat |
| `WHISPER_API_BASE` | Whisper API for video transcription |
| `VLM_API_BASE` / `VLM_MODEL` | Vision LLM for video analysis |
| `DOCLING_API_BASE` / `DOCLING_MODEL` | Document conversion model |
| `DB_*`, `QDRANT_*`, `REDIS_*` | Service connection settings |

## License

MIT
