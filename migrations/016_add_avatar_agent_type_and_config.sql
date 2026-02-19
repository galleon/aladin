-- Migration 016: Add 'avatar' agent type and avatar_config column to agents table

-- Add avatar_config JSON column to agents table (nullable, stores video_source_url/image_url)
ALTER TABLE agents ADD COLUMN IF NOT EXISTS avatar_config JSONB;
