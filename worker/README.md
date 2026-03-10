# Ingestion Pipeline

Document and web content ingestion workers for the RAG Platform.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Main docker-compose.yml                      │
├─────────────────┬─────────────┬─────────────┬─────────────────┤
│  Backend API    │  Redis      │  PostgreSQL │  Qdrant         │
│  /api/ingestion │  (queue)    │             │  (vectors)      │
└────────┬────────┴──────┬──────┴─────────────┴─────────────────┘
         │               │
         │  enqueue      │  dequeue
         ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Ingestion Workers (scalable: 1-100)                 │
│              crawl4ai + docling + langchain text splitters       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

The ingestion worker is included in the main `docker-compose.yml`:

```bash
# Start all services including workers
docker-compose up -d

# Scale workers based on queue depth
docker-compose up -d --scale ingestion-worker=10
```

## API Endpoints

All ingestion endpoints are served by the main backend API at `http://localhost:3000/api/ingestion/`.

### Ingest Web Content

```bash
curl -X POST http://localhost:3000/api/ingestion/web \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "source": {
      "url": "https://investor.nvidia.com/financials",
      "depth_limit": 2,
      "strategy": "bfs",
      "inclusion_patterns": ["*/quarterly-results/*"],
      "exclusion_patterns": ["*/login", "*/legal/*"]
    },
    "processing_config": {
      "render_js": true,
      "wait_for_selector": "div.financial-table",
      "extract_tables": true,
      "table_strategy": "summary_and_html"
    },
    "collection_name": "nvidia_financials"
  }'
```

### Ingest File

```bash
curl -X POST http://localhost:3000/api/ingestion/file \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "collection_name=my_documents" \
  -F "extract_tables=true"
```

### Check Job Status

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:3000/api/ingestion/jobs/{job_id}
```

### List Jobs

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:3000/api/ingestion/jobs?status=completed
```

### Cancel Job

```bash
curl -X DELETE -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:3000/api/ingestion/jobs/{job_id}
```

### List Collections

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:3000/api/ingestion/collections
```

## File Processors

The ingestion worker always uses `RichFileProcessor` — the NeMo-Retriever-style pipeline that produces separate text, table, and image chunk types with bounding boxes and VLM image captions.

### Rich Processor — Qdrant Point Metadata

Every vector point stored by the rich processor carries the following payload fields:

#### Core fields (both processors)

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | The text to embed and return in RAG responses. For tables, a Markdown rendering; for images, the VLM caption. |
| `source_file` | string | Original filename (e.g. `report.pdf`). |
| `chunk_index` | int | 0-based index of this chunk within the document. |
| `total_chunks` | int | Total number of chunks produced for the document. |
| `chunk_id` | string | Unique chunk identifier (`{job_id}_{uuid8}`). |
| `source_type` | string | Always `"file"` for file ingestion. |
| `file_type` | string | Extension without dot (e.g. `"pdf"`, `"docx"`). |
| `page_number` | int \| null | 1-based page number where the chunk originates. Null for non-paginated formats (TXT, JSON). |
| `page` | int \| null | Alias of `page_number` for backward compatibility. |

#### Rich-only fields

| Field | Type | Possible values | Description |
|-------|------|-----------------|-------------|
| `content_type` | string | `"text"` `"structured"` `"image"` | Chunk content category. `text` = prose/headers; `structured` = tables; `image` = VLM-captioned embedded image. |
| `text_type` | string | `"body"` `"header"` `"caption"` `"footnote"` `"list_item"` `"table"` `"picture"` | Fine-grained element type inferred from Docling labels. |
| `text_location` | [l, t, r, b] \| null | Four floats in **page-point coordinates** | Bounding box of the chunk on the page, in Docling's native coordinate system (origin = bottom-left of the page, points = 1/72 inch). `l`=left, `t`=top, `r`=right, `b`=bottom. Null when Docling has no provenance data for the chunk (e.g. plain-text files, or fallback extraction). |
| `page_width` | float \| null | Page width in points | Used by the frontend to normalise `text_location` to a [0,1] relative coordinate for CSS overlay rendering. |
| `page_height` | float \| null | Page height in points | Same as above. A standard A4 portrait page is 595 × 842 pt; US Letter is 612 × 792 pt. |
| `table_content` | string \| null | Markdown table | Raw Markdown rendering of the table (`structured` chunks only). |
| `table_format` | string \| null | `"markdown"` | Format of `table_content` (`structured` chunks only). |
| `image_caption` | string \| null | Free text | VLM-generated description of the embedded image (`image` chunks only). |

#### Coordinate system note

Docling (and PDF) uses a **bottom-left origin**: `b < t` and `(l, b)` is the lower-left corner of the box. The frontend's `BboxIndicator` component converts this to CSS top-left coordinates:

```
css_top    = 1 - (t / page_height)
css_left   = l / page_width
css_width  = (r - l) / page_width
css_height = (t - b) / page_height
```

---

## Supported File Formats

| Format     | Extension       | Processor |
| ---------- | --------------- | --------- |
| PDF        | `.pdf`          | docling   |
| Word       | `.docx`, `.doc` | docling   |
| PowerPoint | `.pptx`, `.ppt` | docling   |
| Markdown   | `.md`           | docling   |
| HTML       | `.html`, `.htm` | docling   |
| Plain Text | `.txt`          | native    |
| JSON       | `.json`         | native    |

## Web Crawling Options

| Option               | Description                                  | Default |
| -------------------- | -------------------------------------------- | ------- |
| `depth_limit`        | Maximum crawl depth                          | 2       |
| `strategy`           | `bfs` (Breadth-First) or `dfs` (Depth-First) | `bfs`   |
| `max_pages`          | Maximum pages to crawl                       | 100     |
| `inclusion_patterns` | URL patterns to include (glob)               | `[]`    |
| `exclusion_patterns` | URL patterns to exclude (glob)               | `[]`    |
| `render_js`          | Execute JavaScript                           | `false` |
| `wait_for_selector`  | CSS selector to wait for                     | `null`  |
| `wait_timeout`       | Wait timeout in seconds                      | 30      |

## Processing Options

| Option           | Description                                   | Default    |
| ---------------- | --------------------------------------------- | ---------- |
| `extract_tables` | Extract tables from documents                 | `true`     |
| `table_strategy` | `raw_html`, `markdown`, or `summary_and_html` | `markdown` |
| `extract_images` | Extract and describe images using VLM         | `false`    |
| `vlm_model`      | VLM model for image description               | `null`     |

## Observability

Enable observability with:

```bash
docker-compose --profile observability up -d
```

View traces in Jaeger UI: http://localhost:16686

### Metrics

| Metric                          | Description               |
| ------------------------------- | ------------------------- |
| `ingestion.documents_processed` | Total documents processed |
| `ingestion.chunks_created`      | Total chunks created      |
| `ingestion.extraction_failures` | Extraction failures       |
| `ingestion.pages_crawled`       | Total pages crawled       |
| `ingestion.processing_duration` | Processing time by stage  |

## Scaling

### Scale Workers

```bash
# Scale to 10 workers
docker-compose up -d --scale ingestion-worker=10
```

### Monitor Queue Depth

```bash
docker exec aladin-redis-1 redis-cli LLEN arq:queue
```

## Standalone Deployment

For running workers separately (e.g., on different machines):

```bash
# Start with standalone Redis/Qdrant
docker-compose -f worker/docker-compose.yml --profile standalone up -d

# Or connect to existing main stack
REDIS_HOST=main-redis QDRANT_HOST=main-qdrant \
  docker-compose -f worker/docker-compose.yml up -d
```

## Environment Variables

| Variable              | Description                | Default                    |
| --------------------- | -------------------------- | -------------------------- |
| `REDIS_HOST`          | Redis hostname             | `redis`                    |
| `REDIS_PORT`          | Redis port                 | `6379`                     |
| `QDRANT_HOST`         | Qdrant hostname            | `qdrant`                   |
| `QDRANT_PORT`         | Qdrant port                | `6333`                     |
| `LLM_API_BASE`        | LLM API endpoint           | `http://localhost:8000/v1` |
| `EMBEDDING_MODEL`     | Embedding model name       | `text-embedding-3-small`   |
| `EMBEDDING_DIMENSION` | Embedding vector size      | `1536`                     |
| `WORKER_CONCURRENCY`  | Jobs per worker            | `5`                        |
| `JOB_TIMEOUT`         | Max job duration (seconds) | `600`                      |
| `CHUNK_SIZE`          | Text chunk size            | `1000`                     |
| `CHUNK_OVERLAP`       | Chunk overlap              | `200`                      |
| `OTEL_ENABLED`        | Enable OpenTelemetry       | `false`                    |

---

## Video ingestion (MP4)

MP4 → time-chunked JSONL for RAG (embedding + Qdrant). Uses fixed-window segmentation, biased-edge frame sampling, optional tracking, and a pluggable VLM backend.

### Quick usage

From the `worker` directory (or with the package installed):

```bash
# Procedure mode (steps: actions, tools, results, warnings)
python -m worker video demo.mp4 --mode procedure --segment-sec 4 --overlap-sec 1 --frames 10 --out chunks.jsonl

# Race mode (per-track commentary, interactions) with simple tracker
python -m worker video race_cam.mp4 --mode race --out chunks.jsonl --tracker simple
```

### Options

| Option          | Description                  | Default        |
| --------------- | ---------------------------- | -------------- |
| `--mode`        | `procedure` or `race`        | `procedure`    |
| `--segment-sec` | Segment window (s)           | 4              |
| `--overlap-sec` | Overlap between segments (s) | 1              |
| `--frames`      | Frames per segment (8–12)    | 10             |
| `--out`         | Output JSONL path            | `chunks.jsonl` |
| `--tracker`     | `noop` or `simple`           | `noop`         |

### Plug in Cosmos VLM

Replace the default `DummyVLMBackend` with `CosmosVLMBackend` in your pipeline:

```python
from worker.video.video_pipeline import run_video_pipeline
from worker.video.vlm_backend import CosmosVLMBackend
from worker.video.tracker import NoopTracker

vlm = CosmosVLMBackend(
    api_base="https://your-cosmos-api.example/v1",
    api_key="YOUR_API_KEY",
    model_id="cosmos-vision-1",
)

run_video_pipeline(
    "video.mp4",
    "chunks.jsonl",
    mode="procedure",
    vlm=vlm,
    tracker=NoopTracker(),
)
```

Implement the HTTP/encoding logic in `CosmosVLMBackend.analyze_segment` (see `worker/video/vlm_backend.py`). For on‑prem, add another backend class that conforms to the `VLMBackend` protocol and pass it as `vlm`.

---

## Unified CLI

All ingestion tasks use a single entry point:

```bash
# File: extract and chunk document to JSONL
python -m worker file document.pdf --out chunks.jsonl

# Video: segment and analyze MP4
python -m worker video demo.mp4 --mode procedure --out chunks.jsonl

# Web: crawl URL and chunk to JSONL
python -m worker web https://example.com --out chunks.jsonl --depth 2 --max-pages 10
```

For full pipeline (embed + Qdrant), use the API.
