-- Migration: Add VLM configuration to data_domains and processing_type to documents
-- Run this migration to add video processing support

-- Add VLM configuration columns to data_domains table
ALTER TABLE data_domains
ADD COLUMN IF NOT EXISTS vlm_api_base VARCHAR(512),
ADD COLUMN IF NOT EXISTS vlm_api_key VARCHAR(512),
ADD COLUMN IF NOT EXISTS vlm_model_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS video_mode VARCHAR(50);

-- Add processing_type column to documents table
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS processing_type VARCHAR(50) DEFAULT 'document';

-- Update existing documents to have 'document' as processing_type
UPDATE documents
SET processing_type = 'document'
WHERE processing_type IS NULL;
