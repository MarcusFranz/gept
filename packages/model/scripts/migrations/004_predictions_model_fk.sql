-- Migration 004: Add model_id FK to predictions table
-- Links each prediction to the model that generated it
--
-- Run: psql -h localhost -U osrs_user -d osrs_data -f scripts/migrations/004_predictions_model_fk.sql

BEGIN;

-- Add model_id column to predictions table
-- Using BIGINT to match model_registry.id
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS model_id BIGINT;

-- Note: We don't add a FK constraint because:
-- 1. predictions is a hypertable with compression
-- 2. FK constraints on hypertables have limitations
-- 3. We validate model_id at application level

-- Index for efficient joins
CREATE INDEX IF NOT EXISTS idx_predictions_model_id
    ON predictions(model_id)
    WHERE model_id IS NOT NULL;

-- Comment
COMMENT ON COLUMN predictions.model_id IS 'Reference to model_registry.id that generated this prediction';

-- View: Predictions with model info
CREATE OR REPLACE VIEW predictions_with_model AS
SELECT
    p.time,
    p.item_id,
    p.item_name,
    p.hour_offset,
    p.offset_pct,
    p.fill_probability,
    p.expected_value,
    p.confidence,
    p.model_id,
    mr.run_id AS model_run_id,
    mr.mean_auc AS model_auc,
    mr.status AS model_status,
    mr.trained_at AS model_trained_at
FROM predictions p
LEFT JOIN model_registry mr ON mr.id = p.model_id
WHERE p.time > NOW() - INTERVAL '1 hour';

COMMENT ON VIEW predictions_with_model IS 'Recent predictions with associated model metadata';

-- Training status table for simple status flags
CREATE TABLE IF NOT EXISTS training_status (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE training_status IS 'Key-value store for training pipeline status flags';

-- Insert default status values
INSERT INTO training_status (key, value) VALUES
    ('last_training_run', '{"run_id": null, "status": "never", "completed_at": null}'::jsonb),
    ('pipeline_health', '{"status": "healthy", "last_check": null, "issues": []}'::jsonb),
    ('next_scheduled_run', '{"scheduled_at": null, "items_queued": 0}'::jsonb)
ON CONFLICT (key) DO NOTHING;

-- Function to update training status
CREATE OR REPLACE FUNCTION update_training_status(
    p_key TEXT,
    p_value JSONB
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO training_status (key, value, updated_at)
    VALUES (p_key, p_value, NOW())
    ON CONFLICT (key) DO UPDATE SET
        value = p_value,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_training_status IS 'Update a training status flag';

COMMIT;
