-- Remove deprecated visual agent and related text detection paths
DELETE FROM code WHERE code LIKE '%visual_agent%' OR code LIKE '%vision_utils.detect_text%';
VACUUM;
