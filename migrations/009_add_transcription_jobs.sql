-- Migration: Add transcription_jobs table for async video transcription
CREATE TABLE IF NOT EXISTS transcription_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    agent_id INTEGER NOT NULL REFERENCES agents(id),

    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    source_filename VARCHAR(255) NOT NULL,
    source_path VARCHAR(512) NOT NULL,
    job_directory VARCHAR(512) NOT NULL,

    language VARCHAR(50),
    subtitle_language VARCHAR(50),
    translation_agent_id INTEGER REFERENCES agents(id),

    result_transcript JSONB,
    result_video_path VARCHAR(512),
    error_message TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS transcription_jobs_user_id_idx ON transcription_jobs(user_id);
CREATE INDEX IF NOT EXISTS transcription_jobs_agent_id_idx ON transcription_jobs(agent_id);
CREATE INDEX IF NOT EXISTS transcription_jobs_status_idx ON transcription_jobs(status);
CREATE INDEX IF NOT EXISTS transcription_jobs_created_at_idx ON transcription_jobs(created_at DESC);
