-- Migration: 004_add_offset_pct_to_outcomes_and_feedback
-- Description: Adds offset_pct attribution columns to outcome + feedback tables
-- Issue: Capture pricing attribution for offset optimization ("option 2")
-- Date: 2026-02-09

ALTER TABLE trade_outcomes
ADD COLUMN IF NOT EXISTS offset_pct NUMERIC(5, 4);

CREATE INDEX IF NOT EXISTS idx_trade_outcomes_offset_pct
ON trade_outcomes(offset_pct);

COMMENT ON COLUMN trade_outcomes.offset_pct IS 'Offset percentage used for the underlying recommendation (0.0125-0.0250)';

ALTER TABLE recommendation_feedback
ADD COLUMN IF NOT EXISTS offset_pct NUMERIC(5, 4);

CREATE INDEX IF NOT EXISTS idx_feedback_offset_pct
ON recommendation_feedback(offset_pct);

COMMENT ON COLUMN recommendation_feedback.offset_pct IS 'Offset percentage used for the underlying recommendation (0.0125-0.0250)';

