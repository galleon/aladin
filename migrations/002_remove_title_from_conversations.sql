-- Migration: Remove title column from conversations table
-- Date: 2025-01-XX
-- Description: Remove title column as we now use metadata.topic exclusively
-- Prerequisite: Ensure all titles are migrated to metadata.topic (done in migration 001)

-- Migrate any remaining titles to metadata.topic (safety check)
UPDATE conversations
SET metadata = jsonb_build_object('topic', title)
WHERE (metadata IS NULL OR metadata::text = 'null' OR metadata::text = '{}')
  AND title IS NOT NULL
  AND title != '';

-- For conversations with no title and no metadata, set default
UPDATE conversations
SET metadata = jsonb_build_object('topic', 'Untitled')
WHERE (metadata IS NULL OR metadata::text = 'null' OR metadata::text = '{}')
  AND (title IS NULL OR title = '');

-- Drop the title column
ALTER TABLE conversations
DROP COLUMN IF EXISTS title;
