-- Allow agents to be linked to several data domains.
-- Run 012_sanity_fix_orphaned_agent_data_domain_id.sql first.
-- 1) Create association table
-- 2) Backfill from current agents.data_domain_id
-- 3) Drop agents.data_domain_id

CREATE TABLE IF NOT EXISTS agent_data_domains (
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    data_domain_id INTEGER NOT NULL REFERENCES data_domains(id) ON DELETE CASCADE,
    PRIMARY KEY (agent_id, data_domain_id)
);

CREATE INDEX IF NOT EXISTS ix_agent_data_domains_data_domain_id ON agent_data_domains(data_domain_id);

-- Backfill: one row per agent that had a single data_domain_id
INSERT INTO agent_data_domains (agent_id, data_domain_id)
SELECT id, data_domain_id FROM agents WHERE data_domain_id IS NOT NULL
ON CONFLICT (agent_id, data_domain_id) DO NOTHING;

-- Drop the single-FK column
ALTER TABLE agents DROP COLUMN IF EXISTS data_domain_id;
