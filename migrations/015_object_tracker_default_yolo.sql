-- Change default object_tracker from 'none' to 'yolo' for new data domains.
-- Existing rows are unchanged.
ALTER TABLE data_domains ALTER COLUMN object_tracker SET DEFAULT 'yolo';
