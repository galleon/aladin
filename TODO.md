# TODO

## Completed (v1 cleanup)

<details>
<summary>All resolved — click to expand</summary>

- [x] Hardcoded chat-ui URLs → configurable via `VITE_CHAT_UI_URL`
- [x] CORS wildcard → configurable via `CORS_ORIGINS`
- [x] Backend pyproject.toml missing dependencies
- [x] Duplicate migration numbering (001 → 003)
- [x] Commented-out model code removed
- [x] Routers `__init__.py` exports all 10 routers
- [x] console.log statements removed from chat-ui
- [x] Duplicate worker Dockerfile clarified
- [x] Config duplication documented
- [x] TODO comment in conversations router fixed
- [x] Hardcoded `spark-9965` hostname replaced
- [x] `datetime.utcnow()` deprecation fixed

</details>

---

## Critical

- [x] **SSRF in web crawler** — Fixed: Added `is_safe_url()` validation that blocks private IP ranges per RFC 1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), loopback addresses, link-local addresses, and cloud metadata endpoints.

- [ ] **JWT tokens stored in localStorage** — `chat-ui/src/store.ts:162` persists JWT to localStorage via Zustand `persist`. Any XSS can steal tokens. Migrate to httpOnly/Secure/SameSite cookies or add token expiration checks client-side.

- [x] **Unsafe `os.environ` mutation in file processor** — Fixed: Added thread-safe lock (`threading.Lock()`) to ensure concurrent workers don't interfere with environment variable manipulation for HybridChunker initialization.

- [x] **Weak default SECRET_KEY** — Fixed: Added startup validation that checks against `DEFAULT_SECRET_KEY` constant and refuses to start in non-development environments with the default key.

## High — Security

- [x] **Path traversal in file processor** — Fixed: Added path validation using `pathlib.Path.resolve()` to ensure all file paths are within `UPLOAD_DIR` before processing.

- [ ] **No rate limiting** — `backend/app/main.py` has no rate limiting on any endpoint. Add `slowapi` or similar, especially on auth endpoints and LLM-calling routes.

- [x] **Missing CSP headers** — Fixed: Added Content-Security-Policy headers to both `frontend/nginx.conf` and `chat-ui/nginx.conf` with additional security headers (X-Content-Type-Options, X-Frame-Options, etc.).

- [ ] **No CSRF protection** — `chat-ui/src/api.ts` sends state-changing requests without CSRF tokens. Add `SameSite=Strict` cookies or `X-CSRF-Token` header.

- [x] **XSS risk in markdown rendering** — Fixed: Added `rehype-sanitize` plugin to all ReactMarkdown components in both frontend and chat-ui.

- [x] **Error messages leak internals** — Fixed: Sanitized error messages in `backend/app/routers/conversations.py` to return generic error messages while logging detailed errors server-side.

## High — Reliability

- [ ] **N+1 queries in conversation listing** — `backend/app/routers/conversations.py:122-124` runs separate queries per conversation for message count and agent details. Use `joinedload()` or subquery aggregation.

- [ ] **N+1 + external call in agent listing** — `backend/app/routers/agents.py:53-54` calls `model_service.get_llm_models()` (external API) on every list request. Cache model list with TTL.

- [x] **No React Error Boundary** — Fixed: Added `ErrorBoundary` component wrapping the entire app in `frontend/src/App.tsx` with graceful error UI.

- [ ] **Zombie jobs — no stuck-job recovery** — `worker/worker/main.py` sets job status to "processing" but has no timeout-based recovery if the worker crashes mid-job. Add a periodic sweep for jobs stuck in "processing" beyond `JOB_TIMEOUT`.

- [x] **Insecure temp file handling** — Fixed: Improved temp file cleanup in `worker/worker/video_processor.py` with robust try-catch error handling in finally block.

## Medium — Backend

- [ ] **Missing pagination on list endpoints** — `backend/app/routers/agents.py`, `data_domains.py`, `conversations.py` load all records into memory. Add `limit`/`offset` query params.

- [ ] **VLM API keys stored in plaintext** — `backend/app/models.py:108-109` stores VLM keys in DB without encryption. Consider a secrets vault or at-rest encryption.

- [x] **Database session leak in worker** — Fixed: Refactored `worker/shared/database.py` to use context manager pattern with automatic session cleanup and rollback on errors.

- [x] **No connection timeout on worker DB** — Fixed: Added `connect_args={"connect_timeout": 10}` to database engine configuration in `worker/shared/database.py`.

- [ ] **Partial embedding failures silently ignored** — `worker/worker/file/processor.py:420-445` sends batches of 100 chunks but doesn't validate response length matches batch size. Log and retry partial failures.

- [ ] **Unvalidated VLM JSON responses** — `worker/worker/video/vlm_backend.py:330-339` parses VLM output with `json.loads()` after regex extraction without schema validation. Use Pydantic model to validate.

## Medium — Frontend

- [ ] **No code splitting** — `frontend/src/App.tsx` imports all page components eagerly. Use `React.lazy()` + `Suspense` for route-based splitting to reduce initial bundle.

- [ ] **Memory leak in polling intervals** — `chat-ui/src/components/TranslationView.tsx:74-102` and `frontend/src/pages/TranscriptionJobs.tsx` set intervals without reliable cleanup on unmount. Use `useEffect` cleanup + `AbortController`.

- [ ] **Race conditions in ChatView state** — `chat-ui/src/components/ChatView.tsx:110-133` uses multiple `setTimeout(0)` to schedule state updates, creating stale closure bugs. Use functional `setState` consistently and remove setTimeout hacks.

- [ ] **Missing error handling in Chat send** — `frontend/src/pages/Chat.tsx:44-59` `sendMessageMutation` has no `onError` handler. Network failures silently swallowed. Add error toast and retry.

- [ ] **Unhandled auth promise rejection** — `frontend/src/App.tsx:50-65` catches auth errors silently with `localStorage.removeItem`. No user notification or cleanup flag for unmounted components.

- [ ] **Missing AbortController on unmount** — `frontend/src/pages/VideoTranscription.tsx:83-98` has cancellation flag but doesn't cancel actual fetch requests. Use `AbortController`.

## Low

- [ ] **Inconsistent error display** — Frontend uses `alert()` in some places (Login, Register) and inline error state in others. Create a centralized toast/notification system.

- [ ] **Missing accessibility (a11y) attributes** — Chat message sources, expand/collapse buttons, and form elements across chat-ui and frontend lack `aria-label`, `aria-expanded`, `role` attributes.

- [ ] **No password strength validation** — `frontend/src/pages/Register.tsx:90` only enforces `minLength={8}`. Add real-time strength indicator.

- [ ] **Missing pagination info in UI** — `frontend/src/pages/DataDomainInspect.tsx` shows "Load more" but no total count or progress indicator.

- [ ] **Hardcoded chat-ui API base URL** — `chat-ui/src/api.ts:1-6` hardcodes `/api`. Read from `import.meta.env.VITE_API_URL` with fallback.

- [ ] **Lazy imports in video tracker** — `worker/worker/video/tracker.py:15-23` imports dependencies inside functions. Errors only surface at runtime. Import at module level for fast-fail.

- [ ] **No embedding dimension validation** — `worker/shared/config.py:58-62` casts `EMBEDDING_DIMENSION` to int without range check. Add `gt=0` validation.

- [ ] **Unused imports** — `frontend/src/pages/AgentDetail.tsx:6` imports `Languages` icon but never uses it.

- [ ] **Magic numbers scattered** — Poll intervals (`4000`, `2000`), max heights (`90vh`), batch sizes (`100`) hardcoded across files. Extract to shared constants.

- [ ] **Excessive API polling** — `frontend/src/pages/TranslationChat.tsx:51-65` polls every 2-10s regardless of tab visibility. Pause polling when tab is hidden.
