-- Migration: 016_prediction_outcomes.sql
-- Purpose: Track prediction accuracy by comparing predictions to actual price outcomes

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id SERIAL PRIMARY KEY,

    -- Link to original prediction
    prediction_time TIMESTAMPTZ NOT NULL,
    item_id INTEGER NOT NULL,
    hour_offset INTEGER NOT NULL,
    offset_pct DECIMAL(5,4) NOT NULL,

    -- What was predicted
    predicted_fill_probability DECIMAL(7,6) NOT NULL,
    predicted_buy_price DECIMAL(12,2) NOT NULL,
    predicted_sell_price DECIMAL(12,2) NOT NULL,

    -- What actually happened
    evaluation_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actual_low_price DECIMAL(12,2),  -- Lowest price in window
    actual_high_price DECIMAL(12,2), -- Highest price in window
    buy_would_fill BOOLEAN,          -- Did price go low enough?
    sell_would_fill BOOLEAN,         -- Did price go high enough?

    -- Classification
    outcome TEXT NOT NULL CHECK (outcome IN (
        'CLEAR_HIT',      -- Low confidence, price never reached (correctly pessimistic)
        'CLEAR_MISS',     -- High confidence, price never reached (wrong)
        'POSSIBLE_HIT',   -- Price reached target
        'POSSIBLE_MISS'   -- Price reached but we predicted very high
    )),

    -- Metadata
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for querying by item and time
CREATE INDEX idx_prediction_outcomes_item_time
ON prediction_outcomes (item_id, prediction_time DESC);

-- Index for aggregating by outcome
CREATE INDEX idx_prediction_outcomes_outcome
ON prediction_outcomes (outcome, created_at DESC);

-- Index for model performance analysis
CREATE INDEX idx_prediction_outcomes_model
ON prediction_outcomes (model_version, outcome);

-- Prevent duplicate evaluations
CREATE UNIQUE INDEX idx_prediction_outcomes_unique
ON prediction_outcomes (prediction_time, item_id, hour_offset, offset_pct);

-- View for quick model health check
CREATE OR REPLACE VIEW prediction_accuracy_summary AS
SELECT
    DATE_TRUNC('day', prediction_time) as prediction_date,
    item_id,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS') as clear_misses,
    COUNT(*) FILTER (WHERE outcome = 'CLEAR_HIT') as clear_hits,
    COUNT(*) FILTER (WHERE outcome = 'POSSIBLE_HIT') as possible_hits,
    COUNT(*) FILTER (WHERE outcome = 'POSSIBLE_MISS') as possible_misses,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as clear_miss_rate_pct
FROM prediction_outcomes
GROUP BY DATE_TRUNC('day', prediction_time), item_id
ORDER BY prediction_date DESC, clear_miss_rate_pct DESC;
