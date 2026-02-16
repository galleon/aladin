-- Migration: Update PostgreSQL enum type to use lowercase values
-- The agenttype enum currently has uppercase values (RAG, TRANSLATION)
-- but the data and Python enum use lowercase (rag, translation)

-- Step 1: Drop the existing enum type (this will fail if it's in use, so we need to change the column first)
-- First, change the column to VARCHAR temporarily
ALTER TABLE agents ALTER COLUMN agent_type TYPE VARCHAR(50) USING agent_type::text;

-- Step 2: Drop the old enum type
DROP TYPE IF EXISTS agenttype CASCADE;

-- Step 3: Create new enum type with lowercase values
CREATE TYPE agenttype AS ENUM ('rag', 'translation', 'video_transcription');

-- Step 4: Change the column back to use the enum type
ALTER TABLE agents ALTER COLUMN agent_type TYPE agenttype USING agent_type::agenttype;

-- Step 5: Verify the enum values
DO $$
DECLARE
    enum_values TEXT;
BEGIN
    SELECT string_agg(enumlabel::text, ', ' ORDER BY enumsortorder) INTO enum_values
    FROM pg_enum
    WHERE enumtypid = (SELECT oid FROM pg_type WHERE typname = 'agenttype');

    RAISE NOTICE 'agenttype enum values: %', enum_values;
END $$;

-- Step 6: Verify all agents have valid enum values
DO $$
DECLARE
    invalid_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO invalid_count
    FROM agents
    WHERE agent_type::text NOT IN ('rag', 'translation', 'video_transcription');

    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'Found % agents with invalid agent_type values', invalid_count;
    ELSE
        RAISE NOTICE 'All agents have valid agent_type values.';
    END IF;
END $$;
