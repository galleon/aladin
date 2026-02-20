-- Add tools column to agents table for dynamic tool-calling agent support.
-- Stores a JSON array of tool names (e.g. ["search_knowledge_base", "translate_text"]).
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tools JSONB NOT NULL DEFAULT '[]'::jsonb;
