-- Migration 017: Ensure composite index on price_data_5min for volume queries
--
-- The engine's liquidity filter queries 24h volume per item:
--   SELECT item_id, SUM(volume) FROM price_data_5min
--   WHERE item_id = ANY(:ids) AND timestamp >= NOW() - INTERVAL '24 hours'
--   GROUP BY item_id
--
-- Without this index, TimescaleDB does a sequential scan across 426M+ rows.
-- The index on (item_id, timestamp DESC) allows an index scan that quickly
-- narrows to the relevant item + time range.

CREATE INDEX IF NOT EXISTS idx_5m_item_ts
    ON price_data_5min (item_id, timestamp DESC);
