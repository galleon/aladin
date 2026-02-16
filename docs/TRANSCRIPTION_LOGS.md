# Transcription job logs

How to inspect logs when video transcription jobs fail or don’t complete.

## Admin UI: all jobs

In the app, go to **Transcription jobs** (sidebar). You’ll see:

- **Type** – Transcribe or Add subtitles
- **Start** – `created_at`
- **End** – `completed_at`
- **Status** – Queued / Processing / Completed / Failed
- **Queue** – Number of jobs waiting in Redis (debug)

If jobs stay in **Queued** and nothing is written under `jobs/transcription/<user_id>/`, the worker is not consuming the queue (see below).

## View logs

**Backend** (queues jobs, enqueue errors):

```bash
docker-compose logs backend
```

**Unified worker** (picks up and runs transcription + ingestion jobs from a single queue):

```bash
docker-compose logs worker
```

**Both, follow mode:**

```bash
docker-compose logs -f backend worker
```

**Last N lines:**

```bash
docker-compose logs --tail=200 worker
```

## Log messages to look for

| Message | Where | Meaning |
|--------|--------|--------|
| `Transcription job queued` | backend | Job created and enqueued; `job_id`, `filename` |
| `Failed to enqueue transcription job` | backend | Redis/ARQ error; job stays `pending` or set `failed` |
| `Transcription job picked from queue` | worker | Worker started processing `job_id` |
| `Processing transcription job` | worker | Pipeline started; `job_id`, `job_type`, `source_filename` |
| `Transcription job not found` | worker | DB has no row for that `job_id` (wrong ID or not committed) |
| `Transcription job already processed` | worker | Job status not `pending`/`processing` (e.g. duplicate run) |
| `Transcription job failed: agent not found` | worker | `agent_id` on job doesn’t exist or was deleted |
| `Transcription job failed: source file not found` | worker | File missing at `source_path`; check `job_directory` and volume mounts |
| `Transcription failed` | worker | Whisper/transcribe step threw; see exception and `job_id` |
| `Subtitle translation failed, using original` | worker | Translation failed; job continues with original SRT |
| `Add subtitles failed` | worker | Burn-in subtitles step failed; see exception |
| `Transcription job completed` | worker | Transcribe-only job finished |
| `Add-subtitles job completed` | worker | Transcribe + translate + burn-in finished |
| `Transcription ready email sent` | backend (email_service) | Notification sent (if email configured) |

## Queue debug

- **Backend:** `GET /api/video-transcription/queue-status` (auth required) returns `queue_name` and `queued_count` for the job queue (`arq:queue`). If you enqueue a job and `queued_count` stays 0, the backend may be enqueueing to a different Redis or DB.
- **Queue-debug:** `GET /api/video-transcription/queue-debug` returns `queued_count`, sample job IDs, `redis_hint`, and `worker_checks` (commands to verify worker service and queue).
- **Worker startup:** Look for `Starting unified worker; queue=arq:queue redis=... db=...` and `Registered tasks: [...]`. The worker must use the same Redis host/port/DB as the backend.

### Redis and logs quick check

```bash
# Is the worker service running? (must be named "worker", not "transcription-worker")
docker compose ps worker

# How many jobs in the queue?
docker compose exec redis redis-cli ZCARD arq:queue

# Which queue is the worker listening to?
docker compose exec worker python3 -c "from app.workers.unified import WorkerSettings; print('queue:', getattr(WorkerSettings, 'queue_name', 'NOT SET'))"

# Worker logs (startup should show queue=arq:queue)
docker compose logs --tail=50 worker

# Rebuild and restart worker (if queue was wrong or code changed)
docker compose up -d --build worker
```

## Jobs stuck in queue (never start)

If jobs stay **Queued** and never move to **Processing**:

1. **Unified worker running and listening to arq:queue?**
   All jobs go to **arq:queue**. The unified worker runs with `arq app.workers.unified.WorkerSettings` and listens to that queue.
   - Check: `docker-compose exec worker python3 -c "from app.workers.unified import WorkerSettings; print('queue_name:', getattr(WorkerSettings, 'queue_name', 'NOT SET'))"`
   - You should see `queue_name: arq:queue`. If the worker was built from an old image, **rebuild and restart**:
     ```bash
     docker-compose build worker && docker-compose up -d worker
     ```

2. **Worker running?**
   ```bash
   docker-compose ps worker
   ```
   If not running, start it: `docker-compose up -d worker`.

3. **Same Redis as backend?**
   Backend and worker must use the same Redis host, port, and DB. Check:
   - Backend: call `GET /api/video-transcription/queue-debug` (see `redis_hint` and `queued_count`).
   - Worker logs: `docker-compose logs worker` and look for:
     `Starting unified worker; queue=arq:queue redis=... db=...`
   - If backend shows e.g. `redis_hint: { host: "redis", port: 6379, db: 0 }` and worker logs show `redis=redis port=6379 db=0`, they match. If one uses `localhost` and the other `redis`, they are different (e.g. backend on host, worker in Docker).

4. **Redis DB**
   In `docker-compose.yml`, both `backend` and `worker` should have `REDIS_DB: "0"` (or the same value). Restart both after changing.

5. **Purge and retry**
   If the queue is stuck, purge from the admin UI (Transcription jobs → Purge queue), then submit new jobs after confirming the worker is running and using the same Redis.

6. **Queue-debug endpoint**
   Call `GET /api/video-transcription/queue-debug` (authenticated). It returns `queued_count`, `job_ids_sample`, `redis_hint`, and `worker_checks`: step-by-step commands to verify the worker service name, queue name, and how to rebuild/restart.

7. **Re-queued jobs never go to "Processing"**
   If you re-queue a job and it stays "Queued" (or briefly appears then goes back to 0 in Redis), the worker is picking it but failing before or during the pipeline. Check worker logs: `docker compose logs worker`. Look for "Dispatching task task_name=transcription", "Transcription job picked from queue", then either "Processing transcription job" (success) or "Transcription pipeline failed" / "Transcription job not found" / "Source file not found". Common causes: worker cannot reach PostgreSQL (DB_HOST, network), worker cannot see the job directory (UPLOAD_DIR, volume mount), or Whisper API unreachable. If the worker fails, the job is now marked "failed" with an error_message so it no longer stays "Queued".

## Common issues

1. **Job stays `pending`**
   Check backend for `Failed to enqueue transcription job` (Redis/ARQ).
   Check that `worker` is running and listening on queue `arq:queue`.

2. **Source file not found**
   Worker runs in container; `source_path` is under `UPLOAD_DIR` (e.g. `/app/uploads/...`).
   Ensure the same `UPLOAD_DIR` and `./jobs` (or equivalent) are mounted in both `backend` and `worker`.

3. **Transcription failed (exception)**
   Usually Whisper API: wrong `WHISPER_API_BASE`/`WHISPER_API_KEY`, network, or file format.
   Check worker env and logs for the exception text.

4. **Agent not found**
   Job references an agent that was deleted or wrong `agent_id`.
   Re-create the job with a valid video-transcription agent.
