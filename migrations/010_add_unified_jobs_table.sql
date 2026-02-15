-- Migration: Add unified jobs table (Redis for queueing only, DB for all job state)
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    user_id INTEGER NOT NULL REFERENCES users(id),
    agent_id INTEGER REFERENCES agents(id),
    document_id INTEGER REFERENCES documents(id),

    payload JSONB NOT NULL,
    result JSONB,

    progress INTEGER DEFAULT 0,
    error_message TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS jobs_job_type_idx ON jobs(job_type);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_user_id_idx ON jobs(user_id);
CREATE INDEX IF NOT EXISTS jobs_agent_id_idx ON jobs(agent_id);
CREATE INDEX IF NOT EXISTS jobs_document_id_idx ON jobs(document_id);
CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at DESC);

COMMENT ON TABLE jobs IS 'Unified job record: queue in Redis, state in DB (transcription, translation, ingestion_web, ingestion_file, ingestion_video)';
COMMENT ON COLUMN jobs.payload IS 'Type-specific input parameters (JSON)';
COMMENT ON COLUMN jobs.result IS 'Type-specific output after completion (JSON)';
