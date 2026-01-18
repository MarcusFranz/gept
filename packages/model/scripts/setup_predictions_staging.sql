-- Staging Table for Testing New Inference Models
-- Database: osrs_data
--
-- This creates a predictions_staging table with identical schema to predictions.
-- Use --staging flag with run_inference.py to write here instead of production.
--
-- Run with: psql -h localhost -U osrs_user -d osrs_data -f setup_predictions_staging.sql

-- Drop and recreate for clean testing (safe since this is staging only)
DROP TABLE IF EXISTS predictions_staging CASCADE;

-- Create staging table with identical schema to predictions
CREATE TABLE predictions_staging (
    id BIGSERIAL,
    time TIMESTAMPTZ NOT NULL,                  -- Prediction generation time
    item_id INTEGER NOT NULL,                   -- OSRS item ID
    item_name TEXT NOT NULL,                    -- Human-readable name
    hour_offset INTEGER NOT NULL,               -- Hours ahead (1-48 for new models)
    target_hour TIMESTAMPTZ NOT NULL,           -- When the prediction is for
    offset_pct DECIMAL(5,4) NOT NULL,           -- Price offset (0.0125 to 0.025)
    fill_probability DECIMAL(7,6) NOT NULL,     -- Model's fill probability [0,1]
    expected_value DECIMAL(8,6) NOT NULL,       -- EV = prob * profit
    buy_price DECIMAL(12,2) NOT NULL,           -- Suggested buy price
    sell_price DECIMAL(12,2) NOT NULL,          -- Suggested sell price
    current_high DECIMAL(12,2),                 -- Current high price
    current_low DECIMAL(12,2),                  -- Current low price
    confidence TEXT NOT NULL DEFAULT 'medium',  -- low/medium/high
    model_version TEXT DEFAULT 'v2-multitarget',-- Model version for tracking
    PRIMARY KEY (time, item_id, hour_offset, offset_pct)
);

-- Convert to hypertable for TimescaleDB compatibility
SELECT create_hypertable('predictions_staging', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Basic indexes for testing queries
CREATE INDEX IF NOT EXISTS idx_staging_latest
    ON predictions_staging(time DESC);

CREATE INDEX IF NOT EXISTS idx_staging_ev
    ON predictions_staging(time DESC, expected_value DESC)
    WHERE expected_value > 0;

-- View for quick inspection
CREATE OR REPLACE VIEW staging_latest AS
SELECT *
FROM predictions_staging
WHERE time = (SELECT MAX(time) FROM predictions_staging);

CREATE OR REPLACE VIEW staging_top AS
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
FROM predictions_staging
WHERE time = (SELECT MAX(time) FROM predictions_staging)
  AND fill_probability >= 0.05
  AND fill_probability < 0.30
ORDER BY expected_value DESC
LIMIT 50;

-- Comparison view: new vs old predictions (when both have data)
CREATE OR REPLACE VIEW staging_vs_production AS
SELECT
    COALESCE(s.item_id, p.item_id) as item_id,
    COALESCE(s.item_name, p.item_name) as item_name,
    COALESCE(s.hour_offset, p.hour_offset) as hour_offset,
    COALESCE(s.offset_pct, p.offset_pct) as offset_pct,
    s.fill_probability as staging_prob,
    p.fill_probability as prod_prob,
    s.expected_value as staging_ev,
    p.expected_value as prod_ev,
    CASE
        WHEN s.fill_probability IS NULL THEN 'prod_only'
        WHEN p.fill_probability IS NULL THEN 'staging_only'
        ELSE 'both'
    END as presence
FROM (SELECT * FROM predictions_staging WHERE time = (SELECT MAX(time) FROM predictions_staging)) s
FULL OUTER JOIN (SELECT * FROM predictions WHERE time = (SELECT MAX(time) FROM predictions)) p
    ON s.item_id = p.item_id
    AND s.hour_offset = p.hour_offset
    AND s.offset_pct = p.offset_pct
ORDER BY COALESCE(s.expected_value, p.expected_value) DESC
LIMIT 100;

COMMENT ON TABLE predictions_staging IS 'Staging table for testing new inference models without affecting production';

SELECT 'Staging table setup complete!' as status;
SELECT 'Run inference with: python run_inference.py --staging' as next_step;
