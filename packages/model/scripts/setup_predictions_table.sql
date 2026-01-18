-- TimescaleDB Schema for GE Flipping Predictions
-- Database: osrs_data (same as source price data)
--
-- Run with: psql -h localhost -U osrs_user -d osrs_data -f setup_predictions_table.sql

-- Ensure TimescaleDB extension is enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Drop existing tables if needed (uncomment if resetting)
-- DROP TABLE IF EXISTS actual_fills CASCADE;
-- DROP TABLE IF EXISTS predictions CASCADE;
-- DROP TABLE IF EXISTS calibration_metrics CASCADE;

-- ============================================================================
-- PREDICTIONS TABLE
-- Stores model predictions, refreshed every 5 minutes
-- ============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL,
    time TIMESTAMPTZ NOT NULL,                  -- Prediction generation time
    item_id INTEGER NOT NULL,                   -- OSRS item ID
    item_name TEXT NOT NULL,                    -- Human-readable name
    hour_offset INTEGER NOT NULL,               -- Hours ahead (1-24)
    target_hour TIMESTAMPTZ NOT NULL,           -- When the prediction is for
    offset_pct DECIMAL(5,4) NOT NULL,           -- Price offset (0.02 = 2%)
    fill_probability DECIMAL(7,6) NOT NULL,     -- Model's fill probability [0,1]
    expected_value DECIMAL(8,6) NOT NULL,       -- EV = prob * profit
    buy_price DECIMAL(12,2) NOT NULL,           -- Suggested buy price
    sell_price DECIMAL(12,2) NOT NULL,          -- Suggested sell price
    current_high DECIMAL(12,2),                 -- Current high price
    current_low DECIMAL(12,2),                  -- Current low price
    confidence TEXT NOT NULL DEFAULT 'medium',  -- low/medium/high
    model_version TEXT DEFAULT 'v1',            -- Model version for tracking
    PRIMARY KEY (time, item_id, hour_offset, offset_pct)
);

-- Convert to hypertable (time-partitioned for performance)
SELECT create_hypertable('predictions', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_predictions_item
    ON predictions(item_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_ev
    ON predictions(time DESC, expected_value DESC)
    WHERE expected_value > 0;

CREATE INDEX IF NOT EXISTS idx_predictions_prob
    ON predictions(time DESC, fill_probability DESC)
    WHERE fill_probability > 0.05;

CREATE INDEX IF NOT EXISTS idx_predictions_latest
    ON predictions(time DESC);

-- ============================================================================
-- ACTUAL FILLS TABLE
-- Tracks whether predictions actually filled (for calibration monitoring)
-- ============================================================================

CREATE TABLE IF NOT EXISTS actual_fills (
    id BIGSERIAL,
    time TIMESTAMPTZ NOT NULL,                  -- When evaluation was performed
    prediction_time TIMESTAMPTZ NOT NULL,       -- Original prediction time
    item_id INTEGER NOT NULL,
    hour_offset INTEGER NOT NULL,
    offset_pct DECIMAL(5,4) NOT NULL,
    predicted_probability DECIMAL(7,6) NOT NULL,
    buy_target DECIMAL(12,2) NOT NULL,          -- The buy price we predicted
    sell_target DECIMAL(12,2) NOT NULL,         -- The sell price we predicted
    actual_min_low DECIMAL(12,2),               -- Actual lowest low in window
    actual_max_high DECIMAL(12,2),              -- Actual highest high in window
    buy_would_fill BOOLEAN NOT NULL,            -- Did buy order fill?
    sell_would_fill BOOLEAN NOT NULL,           -- Did sell order fill?
    both_would_fill BOOLEAN NOT NULL,           -- Did roundtrip complete?
    PRIMARY KEY (time, prediction_time, item_id, hour_offset, offset_pct)
);

SELECT create_hypertable('actual_fills', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fills_prediction
    ON actual_fills(prediction_time, item_id);

CREATE INDEX IF NOT EXISTS idx_fills_calibration
    ON actual_fills(time DESC, predicted_probability);

-- ============================================================================
-- CALIBRATION METRICS TABLE
-- Stores daily calibration summaries for monitoring
-- ============================================================================

CREATE TABLE IF NOT EXISTS calibration_metrics (
    id BIGSERIAL,
    time TIMESTAMPTZ NOT NULL,
    bucket_low DECIMAL(5,4) NOT NULL,
    bucket_high DECIMAL(5,4) NOT NULL,
    prediction_count INTEGER NOT NULL,
    avg_predicted_prob DECIMAL(7,6) NOT NULL,
    actual_fill_rate DECIMAL(7,6) NOT NULL,
    calibration_error DECIMAL(7,6) NOT NULL,
    PRIMARY KEY (time, bucket_low)
);

SELECT create_hypertable('calibration_metrics', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- COMPRESSION POLICIES
-- Automatically compress old data for storage savings
-- ============================================================================

-- Enable compression on predictions
ALTER TABLE predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'item_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Compress chunks older than 7 days
SELECT add_compression_policy('predictions', INTERVAL '7 days', if_not_exists => TRUE);

-- Enable compression on actual_fills
ALTER TABLE actual_fills SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'item_id',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('actual_fills', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- USEFUL VIEWS
-- ============================================================================

-- Latest predictions view
CREATE OR REPLACE VIEW latest_predictions AS
SELECT *
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions);

-- Top opportunities view (filters out broken >30% predictions)
CREATE OR REPLACE VIEW top_opportunities AS
SELECT
    item_id,
    item_name,
    hour_offset,
    offset_pct,
    fill_probability,
    expected_value,
    buy_price,
    sell_price,
    confidence
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions)
  AND fill_probability >= 0.05
  AND fill_probability < 0.30
ORDER BY expected_value DESC
LIMIT 50;

-- Calibration summary view
CREATE OR REPLACE VIEW calibration_summary AS
SELECT
    CASE
        WHEN predicted_probability < 0.01 THEN '0-1%'
        WHEN predicted_probability < 0.05 THEN '1-5%'
        WHEN predicted_probability < 0.10 THEN '5-10%'
        WHEN predicted_probability < 0.20 THEN '10-20%'
        ELSE '20%+'
    END as bucket,
    COUNT(*) as predictions,
    ROUND(AVG(predicted_probability)::numeric, 4) as avg_predicted,
    ROUND(AVG(both_would_fill::int)::numeric, 4) as actual_rate,
    SUM(both_would_fill::int) as fills
FROM actual_fills
WHERE time > NOW() - INTERVAL '7 days'
GROUP BY bucket
ORDER BY bucket;

-- ============================================================================
-- EXAMPLE QUERIES (commented out)
-- ============================================================================

-- Best opportunities right now:
-- SELECT * FROM top_opportunities;

-- 24-hour forecast for item 565:
-- SELECT hour_offset, fill_probability, expected_value, buy_price, sell_price
-- FROM latest_predictions WHERE item_id = 565 ORDER BY hour_offset;

-- Calibration check:
-- SELECT * FROM calibration_summary;

COMMENT ON TABLE predictions IS 'GE flipping model predictions, refreshed every 5 minutes';
COMMENT ON TABLE actual_fills IS 'Tracking actual price movements vs predictions for calibration';
COMMENT ON TABLE calibration_metrics IS 'Daily calibration summaries for monitoring model drift';

SELECT 'Schema setup complete!' as status;
