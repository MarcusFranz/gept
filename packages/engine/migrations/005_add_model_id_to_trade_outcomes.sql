-- Migration: 005_add_model_id_to_trade_outcomes
-- Description: Adds model_id attribution to trade_outcomes for ML calibration and model comparisons
-- Date: 2026-02-09

ALTER TABLE trade_outcomes
ADD COLUMN IF NOT EXISTS model_id TEXT;

CREATE INDEX IF NOT EXISTS idx_trade_outcomes_model_id
ON trade_outcomes(model_id);

COMMENT ON COLUMN trade_outcomes.model_id IS 'Model identifier that generated the underlying recommendation (e.g. patchtst_patchtst, gept_v2)';

