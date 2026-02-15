-- Migration: Add check constraint to ensure agent_type values are lowercase
-- This prevents future case inconsistencies

-- Step 1: Add check constraint to enforce lowercase values
ALTER TABLE agents
ADD CONSTRAINT agents_agent_type_lowercase_check
CHECK (agent_type = LOWER(agent_type));

-- Step 2: Verify the constraint was created
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'agents_agent_type_lowercase_check'
    ) THEN
        RAISE NOTICE 'Check constraint agents_agent_type_lowercase_check created successfully.';
    ELSE
        RAISE EXCEPTION 'Failed to create check constraint.';
    END IF;
END $$;

-- Step 3: Test that the constraint works (should fail)
-- This is commented out as it would cause an error, but demonstrates the constraint
-- INSERT INTO agents (name, owner_id, agent_type) VALUES ('Test', 1, 'RAG'); -- Should fail
