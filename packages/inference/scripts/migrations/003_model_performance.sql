-- Migration 003: Model Performance Tracking Table
-- Rolling performance metrics for drift detection
--
-- Run: psql -h localhost -U osrs_user -d osrs_data -f scripts/migrations/003_model_performance.sql

BEGIN;

-- Performance tracking table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS model_performance (
    time TIMESTAMPTZ NOT NULL,
    model_id BIGINT NOT NULL REFERENCES model_registry(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL,

    -- Performance window
    window_hours INTEGER NOT NULL DEFAULT 24,  -- 24h, 168h (week), etc.

    -- Prediction counts
    predictions_count INTEGER,
    fills_observed INTEGER,                    -- How many predictions had fill outcomes

    -- Calibration metrics
    mean_predicted_prob DECIMAL(5,4),          -- Average predicted probability
    actual_fill_rate DECIMAL(5,4),             -- Actual fill rate
    calibration_error DECIMAL(5,4),            -- |predicted - actual|
    brier_score DECIMAL(5,4),                  -- Mean squared error

    -- Estimated AUC (if we have enough positive/negative examples)
    estimated_auc DECIMAL(5,4),
    auc_confidence TEXT,                       -- 'high', 'medium', 'low' based on sample size

    -- Bucket breakdown (for detailed calibration)
    bucket_stats JSONB,                        -- [{bucket: '0.00-0.05', count: N, predicted: P, actual: A}, ...]

    PRIMARY KEY (time, model_id)
);

-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('model_performance', 'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
    END IF;
END $$;

-- Comments
COMMENT ON TABLE model_performance IS 'Rolling performance metrics for drift detection';
COMMENT ON COLUMN model_performance.calibration_error IS 'Absolute difference between predicted and actual fill rates';
COMMENT ON COLUMN model_performance.window_hours IS 'Performance window size in hours (e.g., 24 for daily)';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_model_performance_model
    ON model_performance(model_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_item
    ON model_performance(item_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_calibration
    ON model_performance(calibration_error DESC)
    WHERE calibration_error IS NOT NULL;

-- View: Latest performance per model
CREATE OR REPLACE VIEW latest_model_performance AS
SELECT DISTINCT ON (mp.model_id)
    mp.model_id,
    mp.item_id,
    mr.item_name,
    mr.status AS model_status,
    mp.time AS performance_time,
    mp.predictions_count,
    mp.mean_predicted_prob,
    mp.actual_fill_rate,
    mp.calibration_error,
    mp.estimated_auc,
    mp.auc_confidence,
    mr.mean_auc AS training_auc,
    CASE
        WHEN mp.estimated_auc IS NOT NULL AND mr.mean_auc IS NOT NULL THEN
            mp.estimated_auc - mr.mean_auc
        ELSE NULL
    END AS auc_drift
FROM model_performance mp
JOIN model_registry mr ON mr.id = mp.model_id
WHERE mp.window_hours = 24  -- Daily metrics
ORDER BY mp.model_id, mp.time DESC;

COMMENT ON VIEW latest_model_performance IS 'Most recent 24h performance metrics per model';

-- View: Models with performance degradation
CREATE OR REPLACE VIEW degraded_models AS
SELECT
    lmp.model_id,
    lmp.item_id,
    lmp.item_name,
    lmp.model_status,
    lmp.training_auc,
    lmp.estimated_auc,
    lmp.auc_drift,
    lmp.calibration_error,
    lmp.predictions_count,
    lmp.performance_time,
    CASE
        WHEN lmp.auc_drift < -0.05 THEN 'AUC degradation'
        WHEN lmp.calibration_error > 0.15 THEN 'Poor calibration'
        ELSE 'Unknown'
    END AS degradation_reason
FROM latest_model_performance lmp
WHERE
    lmp.model_status = 'ACTIVE'
    AND (
        lmp.auc_drift < -0.05                    -- AUC dropped by 5%+
        OR lmp.calibration_error > 0.15          -- Calibration error > 15%
    )
    AND lmp.predictions_count >= 100             -- Enough samples to be confident
ORDER BY
    COALESCE(lmp.auc_drift, 0) ASC,
    lmp.calibration_error DESC;

COMMENT ON VIEW degraded_models IS 'Active models showing performance degradation needing retraining';

-- Function: Record performance snapshot
CREATE OR REPLACE FUNCTION record_model_performance(
    p_model_id BIGINT,
    p_item_id INTEGER,
    p_window_hours INTEGER DEFAULT 24
)
RETURNS VOID AS $$
DECLARE
    v_start_time TIMESTAMPTZ;
    v_predictions_count INTEGER;
    v_fills_observed INTEGER;
    v_mean_predicted DECIMAL(5,4);
    v_actual_fill_rate DECIMAL(5,4);
BEGIN
    v_start_time := NOW() - (p_window_hours || ' hours')::INTERVAL;

    -- Calculate metrics from actual_fills table if it exists
    -- This is a placeholder - actual implementation depends on how fills are tracked
    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE actual_filled = TRUE),
        AVG(predicted_probability),
        AVG(CASE WHEN actual_filled THEN 1.0 ELSE 0.0 END)::DECIMAL(5,4)
    INTO
        v_predictions_count,
        v_fills_observed,
        v_mean_predicted,
        v_actual_fill_rate
    FROM actual_fills
    WHERE
        item_id = p_item_id
        AND prediction_time >= v_start_time;

    -- Insert performance record
    INSERT INTO model_performance (
        time, model_id, item_id, window_hours,
        predictions_count, fills_observed,
        mean_predicted_prob, actual_fill_rate,
        calibration_error
    ) VALUES (
        NOW(), p_model_id, p_item_id, p_window_hours,
        v_predictions_count, v_fills_observed,
        v_mean_predicted, v_actual_fill_rate,
        ABS(COALESCE(v_mean_predicted, 0) - COALESCE(v_actual_fill_rate, 0))
    )
    ON CONFLICT (time, model_id) DO UPDATE SET
        predictions_count = EXCLUDED.predictions_count,
        fills_observed = EXCLUDED.fills_observed,
        mean_predicted_prob = EXCLUDED.mean_predicted_prob,
        actual_fill_rate = EXCLUDED.actual_fill_rate,
        calibration_error = EXCLUDED.calibration_error;

END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION record_model_performance IS 'Record performance snapshot for a model';

COMMIT;
