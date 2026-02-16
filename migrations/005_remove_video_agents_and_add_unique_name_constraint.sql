-- Migration: Remove all video transcription agents and add unique constraint on (owner_id, name)
-- This ensures that agents with the same owner cannot have duplicate names

-- Step 1: Delete all video transcription agents
DELETE FROM agents WHERE agent_type = 'VIDEO_TRANSCRIPTION';

-- Step 2: Check for duplicate agent names per owner before adding constraint
-- This will help identify any existing duplicates
DO $$
DECLARE
    duplicate_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO duplicate_count
    FROM (
        SELECT owner_id, name, COUNT(*) as cnt
        FROM agents
        GROUP BY owner_id, name
        HAVING COUNT(*) > 1
    ) duplicates;

    IF duplicate_count > 0 THEN
        RAISE NOTICE 'Found % duplicate agent names. Please resolve duplicates before applying unique constraint.', duplicate_count;
        -- For now, we'll keep the first occurrence and delete duplicates
        -- This keeps the oldest agent (lowest ID) for each (owner_id, name) pair
        DELETE FROM agents
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                       ROW_NUMBER() OVER (PARTITION BY owner_id, name ORDER BY id) as rn
                FROM agents
            ) ranked
            WHERE rn > 1
        );
        RAISE NOTICE 'Removed duplicate agents, keeping the oldest (lowest ID) for each (owner_id, name) pair.';
    END IF;
END $$;

-- Step 3: Add unique constraint on (owner_id, name)
-- This ensures that each owner can only have one agent with a given name
CREATE UNIQUE INDEX IF NOT EXISTS agents_owner_id_name_unique
ON agents(owner_id, name);

-- Verify the constraint was created
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'agents_owner_id_name_unique'
    ) THEN
        RAISE NOTICE 'Unique constraint agents_owner_id_name_unique created successfully.';
    ELSE
        RAISE EXCEPTION 'Failed to create unique constraint.';
    END IF;
END $$;
