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

- [ ] **SSRF in web crawler** — `worker/worker/web/crawler.py` accepts any user-supplied URL without validation. An attacker could target internal IPs (127.0.0.1, 169.254.169.254 metadata endpoints, 10.x private ranges). Block private IP ranges and cloud metadata endpoints.

- [ ] **JWT tokens stored in localStorage** — `chat-ui/src/store.ts:162` persists JWT to localStorage via Zustand `persist`. Any XSS can steal tokens. Migrate to httpOnly/Secure/SameSite cookies or add token expiration checks client-side.

- [ ] **Unsafe `os.environ` mutation in file processor** — `worker/worker/file/processor.py:108-124` modifies global `os.environ` to set OpenAI credentials for HybridChunker. Not thread-safe — concurrent workers could use wrong API keys. Pass credentials as constructor params instead.

- [ ] **Weak default SECRET_KEY** — `backend/app/config.py:35` defaults to `"your-secret-key-change-in-production-min-32-chars"`. Add a startup check that refuses to start with the default key when `ENVIRONMENT != development`.

## High — Security

- [ ] **Path traversal in file processor** — `worker/worker/file/processor.py:185-187` uses `file_path` directly without validating it's within `UPLOAD_DIR`. Use `pathlib.Path.resolve()` and verify prefix.

- [ ] **No rate limiting** — `backend/app/main.py` has no rate limiting on any endpoint. Add `slowapi` or similar, especially on auth endpoints and LLM-calling routes.

- [ ] **Missing CSP headers** — Neither `frontend/nginx.conf` nor `chat-ui/nginx.conf` set Content-Security-Policy headers. Add CSP to prevent XSS/injection.

- [ ] **No CSRF protection** — `chat-ui/src/api.ts` sends state-changing requests without CSRF tokens. Add `SameSite=Strict` cookies or `X-CSRF-Token` header.

- [ ] **XSS risk in markdown rendering** — `frontend/src/pages/DataDomainDetail.tsx:306-322` uses ReactMarkdown without `rehype-sanitize`. User-provided VLM prompts could contain malicious content.

- [ ] **Error messages leak internals** — `backend/app/routers/conversations.py:546-564` exposes raw exception strings (API keys, model names, config) to clients. Log details server-side, return generic messages.

## High — Reliability

- [ ] **N+1 queries in conversation listing** — `backend/app/routers/conversations.py:122-124` runs separate queries per conversation for message count and agent details. Use `joinedload()` or subquery aggregation.

- [ ] **N+1 + external call in agent listing** — `backend/app/routers/agents.py:53-54` calls `model_service.get_llm_models()` (external API) on every list request. Cache model list with TTL.

- [ ] **No React Error Boundary** — `frontend/src/App.tsx` has no Error Boundary. Any runtime error crashes the entire app with a blank screen. Wrap routes in an Error Boundary with fallback UI.

- [ ] **Zombie jobs — no stuck-job recovery** — `worker/worker/main.py` sets job status to "processing" but has no timeout-based recovery if the worker crashes mid-job. Add a periodic sweep for jobs stuck in "processing" beyond `JOB_TIMEOUT`.

- [ ] **Insecure temp file handling** — `worker/worker/video_processor.py:168-171` uses `delete=False` without guaranteed cleanup. If cleanup fails, temp files accumulate. Use context manager with `delete=True` or worker-specific temp directory.

## Medium — Backend

- [ ] **Missing pagination on list endpoints** — `backend/app/routers/agents.py`, `data_domains.py`, `conversations.py` load all records into memory. Add `limit`/`offset` query params.

- [ ] **VLM API keys stored in plaintext** — `backend/app/models.py:108-109` stores VLM keys in DB without encryption. Consider a secrets vault or at-rest encryption.

- [ ] **Database session leak in worker** — `worker/shared/database.py:29-31` returns session without enforcing close. Callers can forget `session.close()`. Return a context manager instead.

- [ ] **No connection timeout on worker DB** — `worker/shared/database.py:18-23` uses `NullPool` with no query timeout. Add `connect_args={"connect_timeout": 10}` and statement timeout.

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
