-- Predictions Table Schema
-- Contract: Model writes, Engine reads
--
-- This is the source of truth for the predictions interface.
-- Any changes here require coordination between model and engine teams.

CREATE TABLE IF NOT EXISTS predictions (
    -- Composite primary key
    time TIMESTAMPTZ NOT NULL,
    item_id INTEGER NOT NULL,
    hour_offset INTEGER NOT NULL,           -- 1-48 hours (18 windows)
    offset_pct DECIMAL(5,4) NOT NULL,       -- 0.0125-0.025 (6 offsets)

    -- Prediction outputs
    fill_probability DECIMAL(7,6) NOT NULL, -- [0, 1] clipped
    expected_value DECIMAL(8,6) NOT NULL,   -- EV = prob * (2 * offset - 0.02)

    -- Price targets
    buy_price DECIMAL(12,2),
    sell_price DECIMAL(12,2),
    current_high DECIMAL(12,2),
    current_low DECIMAL(12,2),

    -- Price stability fields (for anti-manipulation filtering)
    median_14d DECIMAL(12,2),              -- 14-day median price
    price_vs_median_ratio DECIMAL(6,4),    -- current_mid / median_14d

    -- Momentum fields (for anti-adverse-selection filtering)
    return_1h DECIMAL(6,4),                -- 1-hour price return
    return_4h DECIMAL(6,4),                -- 4-hour price return
    return_24h DECIMAL(6,4),               -- 24-hour price return
    volatility_24h DECIMAL(6,4),           -- 24-hour rolling volatility

    -- Metadata
    item_name TEXT,
    confidence TEXT,                         -- low | medium | high
    model_version TEXT,

    PRIMARY KEY (time, item_id, hour_offset, offset_pct)
);

-- Index for freshness queries
CREATE INDEX IF NOT EXISTS idx_predictions_time
ON predictions (time DESC);

-- Index for item lookups
CREATE INDEX IF NOT EXISTS idx_predictions_item
ON predictions (item_id, time DESC);

-- Index for recommendation queries
CREATE INDEX IF NOT EXISTS idx_predictions_ev
ON predictions (time, expected_value DESC)
WHERE fill_probability BETWEEN 0.03 AND 0.50;

-- TimescaleDB hypertable (if using TimescaleDB)
-- SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

COMMENT ON TABLE predictions IS 'ML model predictions for GE flip recommendations';
COMMENT ON COLUMN predictions.hour_offset IS 'Time horizon in hours (1-48)';
COMMENT ON COLUMN predictions.offset_pct IS 'Price offset percentage (0.0125 = 1.25%)';
COMMENT ON COLUMN predictions.fill_probability IS 'Probability both buy and sell orders fill within time window';
COMMENT ON COLUMN predictions.expected_value IS 'Expected value accounting for 2% GE tax';
