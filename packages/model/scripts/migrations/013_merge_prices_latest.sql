-- Migration: Merge prices_latest into prices_latest_1m
-- Issue: #153
-- Date: 2026-01-18
--
-- This migration consolidates two duplicate 1-minute price tables:
-- - prices_latest_1m: 63M rows, actively collecting
-- - prices_latest: 20M rows, stale since 2026-01-10
--
-- Expected outcome: Single table, ~3.5GB disk recovered

-- 1. Check current state (run manually first to verify)
-- SELECT
--     'prices_latest_1m' as tbl, min(timestamp), max(timestamp), count(*)
-- FROM prices_latest_1m
-- UNION ALL
-- SELECT
--     'prices_latest' as tbl, min(timestamp), max(timestamp), count(*)
-- FROM prices_latest;

-- 2. Insert non-overlapping data from prices_latest into prices_latest_1m
-- Note: prices_latest uses high_price/low_price, prices_latest_1m uses high/low
INSERT INTO prices_latest_1m (timestamp, item_id, high, high_time, low, low_time)
SELECT timestamp, item_id, high_price, high_time, low_price, low_time
FROM prices_latest pl
WHERE NOT EXISTS (
    SELECT 1 FROM prices_latest_1m p1m
    WHERE p1m.item_id = pl.item_id
    AND p1m.timestamp = pl.timestamp
)
ON CONFLICT DO NOTHING;

-- 3. Stop the collector service before dropping the table
-- Run on Ampere: docker stop osrs-latest-1m (if it was writing to prices_latest)

-- 4. Drop the old table
DROP TABLE IF EXISTS prices_latest;

-- 5. Vacuum to reclaim space
VACUUM ANALYZE prices_latest_1m;
