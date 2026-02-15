# Agents: Prompts, Model Filtering, and Ingestion

## Summary

- **Prompts**: Exposed and editable for RAG (already) and Translation (added). Shown on AgentDetail with Edit for owners.
- **Model filtering**: LLM list used by agents now excludes embedding, docling, and reranker models. Embedding stays only in Data Domain create; reranker/docling remain ingestion/env-only.
- **Ingestion**: Embedding is chosen when creating a **Data Domain**. Reranker and docling are not in the main app UI; they are configured via ingestion worker environment (e.g. `DOCLING_MODEL`).

---

## 1. Prompts

### RAG agents

- **Create (Agents.tsx)**: `system_prompt` is required and editable (textarea with default).
- **Detail (AgentDetail.tsx)**: Shown in “System Prompt” section. Owner can **Edit** → change text → **Save** (or **Cancel**). Update goes through `PUT /agents/:id` with `system_prompt`.
- **Usage**: `rag_service` uses `agent.system_prompt` in the RAG chain.

### Translation agents

- **Create (Agents.tsx)**: Optional “System prompt (optional)” textarea. Placeholder: *Leave empty to use the default. Use `{target_language}` and `{simplified}` as placeholders.*
- **Detail (AgentDetail.tsx)**: “System Prompt” block shown for Translation too. If `system_prompt` is null/empty: displays *No custom prompt (using built-in default).* Owner can **Edit** to set or change it.
- **Backend**: `TranslationAgentCreate` has `system_prompt: str | None = None`. `translation_service.get_system_prompt()` and `translate_markdown` use `agent.system_prompt` when it is non‑empty; otherwise the built‑in prompt. Custom prompt supports `{target_language}` and `{simplified}`.

### Video transcription agents

- No LLM; no user‑facing system prompt in the main app. SRT translation uses a fixed prompt in `video_transcription_service`; that could be made configurable later if needed.

---

## 2. Model filtering

### Where models are chosen

| Context                | API / source           | Models shown                          |
|------------------------|------------------------|---------------------------------------|
| **RAG agent create**   | `GET /models/llm`      | LLM only (filtered, see below)        |
| **Translation create** | `GET /models/llm`      | LLM only (filtered)                   |
| **Video create**       | —                      | No model choice (Whisper only)        |
| **Data Domain create** | `GET /models/embedding`| Embedding only (`EMBEDDING_API_BASE`) |
| **Ingestion pipeline** | Env / worker config    | `EMBEDDING_MODEL`, `DOCLING_MODEL`    |

### LLM filter (agent create)

`model_service.get_llm_models()` filters out model ids that match any of:

- `embedding`, `embed-`, `bge-`, `e5-`, `nomic-embed`, `text-embedding`
- `docling`, `granite-docling`
- `rerank`, `reranker`, `bge-reranker`, `bge-reranker-`

So the “LLM Model” dropdown in agent create no longer includes typical embedding, docling, or reranker models, even if the LLM endpoint exposes them.

### Embedding, reranker, docling

- **Embedding**: Chosen only when creating a **Data Domain** (`/models/embedding` → `EMBEDDING_API_BASE`). Not in agent create.
- **Reranker**: Not used in the current main app or ingestion code. No UI. If added later, it should live in ingestion/pipeline configuration, not in agent create.
- **Docling**: Used in the **ingestion worker** via `DOCLING_MODEL` (and related config). No main‑app UI; that stays in the ingestion pipeline / env.

---

## 3. Ingestion pipeline and Data Domains

- **Data Domain create** (DataDomains.tsx): user picks `embedding_model` from `modelsApi.listEmbedding()`. That value is stored on the Data Domain and used for how that domain’s data is embedded/indexed.
- **Ingestion worker**: Uses `EMBEDDING_API_BASE`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, and `DOCLING_MODEL` from its own config. How this aligns with the Data Domain’s `embedding_model` is deployment‑specific; the important part for this analysis is that **embedding (and docling) are not selectable when creating agents**.
- **Reranker / docling**: Only in ingestion/env. To “choose them when setting up the ingestion pipeline,” that would be done via ingestion config or a future “Ingestion pipeline settings” UI, not in agent or Data Domain create.

---

## 4. Implemented changes (quick ref)

- **Backend**
  - `model_service.get_llm_models()`: exclude embedding / docling / reranker‑like ids.
  - `TranslationAgentCreate.system_prompt`, `create_translation_agent` sets `agent.system_prompt`.
  - `translation_service`: `_format_custom_prompt()`, `get_system_prompt()` and `translate_markdown()` use `agent.system_prompt` when non‑empty; placeholders `{target_language}`, `{simplified}`.
- **Frontend**
  - **Agents.tsx**: Translation create has optional “System prompt” textarea; `createTranslation` sends `system_prompt`.
  - **AgentDetail.tsx**: “System Prompt” for RAG and Translation; owner can Edit/Save/Cancel; `agentsApi.update(..., { system_prompt })`.
- **API**
  - `createTranslation` in `client.ts` accepts `system_prompt?: string | null`.

---

## 5. Possible follow‑ups

- **RAG AgentDetail**: Already supports editing `system_prompt`; no extra work for “expose and make it customizable.”
- **Translation**: Optional per‑language or “simplified vs normal” overrides could be added on top of the single custom `system_prompt` and `{target_language}` / `{simplified}`.
- **Video / SRT**: If SRT translation is exposed in the UI, its prompt could be made configurable on the video agent.
- **Ingestion UI**: A dedicated “Ingestion pipeline settings” (or similar) could expose docling and, if added, reranker, while keeping them separate from agent and Data Domain create.
- **LLM filter**: If your LLM endpoint exposes non‑LLM models with other id patterns, extend `_EXCLUDE_PATTERNS` in `model_service.py` or make it configurable (e.g. via settings).
