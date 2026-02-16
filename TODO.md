# TODO — Pre-commit cleanup

## High priority

- [x] **Hardcoded chat-ui URLs** — `localhost:7860` is hardcoded in three frontend files. Make it configurable via `VITE_CHAT_UI_URL` env var.
  - `frontend/src/pages/AgentDetail.tsx:87`
  - `frontend/src/pages/Agents.tsx:183`
  - `frontend/src/pages/Dashboard.tsx:157`

- [x] **CORS wildcard** — `backend/app/main.py:55` uses `allow_origins=["*"]`. Restrict to actual frontend origins or make configurable via env var (`CORS_ORIGINS`).

- [x] **Backend pyproject.toml missing dependencies** — Production dependencies are now declared in `pyproject.toml`.

- [x] **Duplicate migration numbering** — Renamed `001_add_vlm_and_processing_type.sql` → `003_add_vlm_and_processing_type.sql`.

## Medium priority

- [x] **Commented-out model code** — Removed commented-out ChatSession (lines 374-387), alias (line 414), and RAGCitation (lines 478-483) from `backend/app/models.py`.

- [x] **Incomplete routers __init__.py** — `backend/app/routers/__init__.py` now exports all 10 routers.

- [x] **console.log statements in chat-ui** — Removed all debug console.log calls from `store.ts`, `ChatView.tsx`, and `TranslationView.tsx`.

- [x] **Duplicate worker Dockerfile** — Added clarifying comments to both `worker.Dockerfile` (root, canonical for docker-compose) and `worker/worker/Dockerfile` (standalone).

- [x] **Config duplication** — Added comment in `worker/shared/config.py` explaining it's intentionally separate from `backend/app/config.py`.

## Low priority

- [x] **TODO comment** — `backend/app/routers/conversations.py:57` — Now loads feedback from the message relationship.

- [x] **Hardcoded `spark-9965` hostname** — Replaced with generic `llm-host` placeholder in `docker-compose.yml` and `env.local.template`.

- [x] **`datetime.utcnow()` deprecation** — Replaced with `datetime.now(timezone.utc)` in `backend/app/services/auth.py`, `backend/app/models.py`, and `backend/app/routers/conversations.py`.
