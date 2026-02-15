-- Sanity check: clear agent.data_domain_id when the referenced data domain no longer exists.
-- Run after any manual DB changes or to fix legacy orphaned FKs.

UPDATE agents
SET data_domain_id = NULL
WHERE data_domain_id IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM data_domains WHERE data_domains.id = agents.data_domain_id);
