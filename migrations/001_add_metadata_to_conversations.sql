-- Migration: Add metadata column to conversations table
-- Date: 2025-01-XX
-- Description: Add OpenAI-compatible metadata JSON column to store topic and other metadata

-- Add metadata column (nullable, defaults to NULL)
ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS metadata JSON;

-- Migrate existing titles to metadata.topic for existing conversations
-- This ensures backward compatibility and OpenAI format compliance
UPDATE conversations
SET metadata = jsonb_build_object('topic', title)
WHERE metadata IS NULL
  AND title IS NOT NULL
  AND title != '';

-- For conversations with no title, set default metadata
UPDATE conversations
SET metadata = jsonb_build_object('topic', 'Untitled')
WHERE metadata IS NULL
  AND (title IS NULL OR title = '');
