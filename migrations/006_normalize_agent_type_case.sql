-- Migration: Normalize agent_type values to lowercase
-- The AgentType enum expects lowercase values: "rag", "translation", "video_transcription"
-- But some agents have uppercase values like "RAG" and "TRANSLATION"

-- Step 1: Normalize all agent_type values to lowercase
UPDATE agents SET agent_type = LOWER(agent_type);

-- Step 2: Verify all values are now lowercase
DO $$
DECLARE
    invalid_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO invalid_count
    FROM agents
    WHERE agent_type NOT IN ('rag', 'translation', 'video_transcription');

    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'Found % agents with invalid agent_type values after normalization', invalid_count;
    ELSE
        RAISE NOTICE 'All agent_type values normalized successfully.';
    END IF;
END $$;

-- Step 3: Show final distribution
SELECT agent_type, COUNT(*) as count
FROM agents
GROUP BY agent_type
ORDER BY agent_type;
