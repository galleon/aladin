-- Migration: Add video pipeline config: vlm_prompt, object_tracker, enable_ocr
-- Run this migration to support customizable VLM prompts, tracker choice, and OCR

ALTER TABLE data_domains
ADD COLUMN IF NOT EXISTS vlm_prompt TEXT,
ADD COLUMN IF NOT EXISTS object_tracker VARCHAR(50) DEFAULT 'none',
ADD COLUMN IF NOT EXISTS enable_ocr BOOLEAN DEFAULT false;
