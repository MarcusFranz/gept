-- Migration 019: Model validation system tables
--
-- Stores:
-- 1) row-level prediction outcomes vs realized market movement
-- 2) aggregated model validation snapshots with development signals

BEGIN;

CREATE TABLE IF NOT EXISTS model_validation_prediction_outcomes (
    prediction_time TIMESTAMPTZ NOT NULL,
    item_id INTEGER NOT NULL,
    hour_offset INTEGER NOT NULL,
    offset_pct DECIMAL(5,4) NOT NULL,
    model_id TEXT NOT NULL DEFAULT 'unknown',

    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    predicted_fill_probability DECIMAL(7,6) NOT NULL,
    predicted_expected_value DECIMAL(8,6) NOT NULL,
    predicted_buy_price DECIMAL(12,2) NOT NULL,
    predicted_sell_price DECIMAL(12,2) NOT NULL,

    actual_low_price DECIMAL(12,2),
    actual_high_price DECIMAL(12,2),
    buy_would_fill BOOLEAN NOT NULL,
    sell_would_fill BOOLEAN NOT NULL,
    realized_fill BOOLEAN NOT NULL,
    realized_edge DECIMAL(8,6),

    is_recommendation_candidate BOOLEAN NOT NULL,
    recommendation_success BOOLEAN,

    PRIMARY KEY (prediction_time, item_id, hour_offset, offset_pct, model_id)
);

CREATE INDEX IF NOT EXISTS idx_mvpo_model_time
    ON model_validation_prediction_outcomes(model_id, prediction_time DESC);

CREATE INDEX IF NOT EXISTS idx_mvpo_recommendation_success
    ON model_validation_prediction_outcomes(recommendation_success, prediction_time DESC)
    WHERE is_recommendation_candidate = TRUE;

COMMENT ON TABLE model_validation_prediction_outcomes
    IS 'Prediction-level validation outcomes computed from realized market data';
COMMENT ON COLUMN model_validation_prediction_outcomes.realized_edge
    IS 'Realized spread edge from observed low/high over horizon, net of 2% GE tax';

CREATE TABLE IF NOT EXISTS model_validation_snapshots (
    id BIGSERIAL PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    model_id TEXT NOT NULL,

    total_evaluated INTEGER NOT NULL,
    recommendation_candidates INTEGER NOT NULL,

    mean_predicted_fill_probability DECIMAL(7,6),
    realized_fill_rate DECIMAL(7,6),
    calibration_error DECIMAL(7,6),
    brier_score DECIMAL(7,6),

    recommendation_success_rate DECIMAL(7,6),
    expected_value_mean DECIMAL(8,6),
    realized_edge_mean DECIMAL(8,6),
    expected_minus_realized_edge DECIMAL(8,6),

    trade_outcomes_count INTEGER NOT NULL DEFAULT 0,
    trade_outcomes_win_rate DECIMAL(7,6),
    trade_outcomes_avg_profit DECIMAL(14,2),
    feedback_count INTEGER NOT NULL DEFAULT 0,
    feedback_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,

    horizon_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
    offset_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
    development_signals JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_mvs_model_generated
    ON model_validation_snapshots(model_id, generated_at DESC);

COMMENT ON TABLE model_validation_snapshots
    IS 'Model-level validation snapshots with aggregated metrics and development signals';
COMMENT ON COLUMN model_validation_snapshots.development_signals
    IS 'Machine-readable guidance for model iteration based on validation metrics';

COMMIT;
