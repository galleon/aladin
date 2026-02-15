# TODO — Pre-commit cleanup

## High priority

- [ ] **Hardcoded chat-ui URLs** — `localhost:7860` is hardcoded in three frontend files. Make it configurable via `VITE_CHAT_UI_URL` env var.
  - `frontend/src/pages/AgentDetail.tsx:87`
  - `frontend/src/pages/Agents.tsx:183`
  - `frontend/src/pages/Dashboard.tsx:157`

- [ ] **CORS wildcard** — `backend/app/main.py:55` uses `allow_origins=["*"]`. Restrict to actual frontend origins or make configurable via env var.

- [ ] **Backend pyproject.toml missing dependencies** — Production dependencies are only declared in the Dockerfile `pip install` commands, not in `pyproject.toml`. The project isn't installable as a Python package outside Docker.

- [ ] **Duplicate migration numbering** — Two migration files share the `001_` prefix: `001_add_metadata_to_conversations.sql` and `001_add_vlm_and_processing_type.sql`. Renumber to establish correct ordering.

## Medium priority

- [ ] **Commented-out model code** — `backend/app/models.py` has large blocks of commented-out ChatSession (lines 374-387) and RAGCitation (lines 478-483) models. Clean up or move to a migration plan doc.

- [ ] **Incomplete routers __init__.py** — `backend/app/routers/__init__.py` only exports 4 of 10 routers. Either export all or remove the file (main.py imports directly anyway).

- [ ] **console.log statements in chat-ui** — `chat-ui/src/` has ~20+ debug console.log calls across `store.ts`, `ChatView.tsx`, and `TranslationView.tsx`. Strip before release.

- [ ] **Duplicate worker Dockerfile** — Both `worker.Dockerfile` (root, used by docker-compose) and `worker/worker/Dockerfile` exist with overlapping purpose. Consolidate or clarify which is canonical.

- [ ] **Config duplication** — `backend/app/config.py` and `worker/shared/config.py` define overlapping settings independently. Consider sharing or documenting the divergence.

## Low priority

- [ ] **TODO comment** — `backend/app/routers/conversations.py:57` — `feedback=None, # TODO: Load feedback if needed`

- [ ] **Hardcoded `spark-9965` hostname** — `docker-compose.yml` and `env.local.template` reference a machine-specific hostname (`spark-9965.local`). Replace with a generic placeholder or document that users must change it.

- [ ] **`datetime.utcnow()` deprecation** — `backend/app/services/auth.py:40,42` uses `datetime.utcnow()` which is deprecated in Python 3.12+. Use `datetime.now(timezone.utc)` instead.
