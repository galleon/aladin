-- Remove agent_data_domains rows that reference deleted data domains.
-- Run if you see agents still linked to non-existing domains.

DELETE FROM agent_data_domains
WHERE data_domain_id NOT IN (SELECT id FROM data_domains);
