-- Migration 018: Materialized views for pre-computed volume aggregates
--
-- Replaces expensive per-request CTEs in the engine's prediction_loader.py
-- that aggregate price_data_5min over 24h/1h windows on every API call.
-- Refreshed every 5 minutes by the inference cron script.
--
-- Uses CONCURRENTLY-compatible unique indexes so refreshes don't block reads.

-- 24-hour volume per item
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_volume_24h AS
SELECT
    item_id,
    COALESCE(SUM(high_price_volume), 0)
    + COALESCE(SUM(low_price_volume), 0) AS total_volume
FROM price_data_5min
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY item_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_volume_24h_item
    ON mv_volume_24h (item_id);

-- 1-hour volume per item
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_volume_1h AS
SELECT
    item_id,
    COALESCE(SUM(high_price_volume), 0)
    + COALESCE(SUM(low_price_volume), 0) AS total_volume
FROM price_data_5min
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY item_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_volume_1h_item
    ON mv_volume_1h (item_id);
