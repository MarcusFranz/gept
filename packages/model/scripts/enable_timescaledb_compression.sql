-- =============================================================================
-- Enable TimescaleDB Compression on price_data_5min
-- =============================================================================
-- This script adds a compression policy and manually compresses existing chunks.
-- Expected to reduce disk usage from 72GB to ~10-15GB.
-- =============================================================================

-- Add compression policy (compress chunks older than 7 days)
-- This runs automatically going forward
SELECT add_compression_policy('price_data_5min', INTERVAL '7 days', if_not_exists => TRUE);

-- Manually compress existing old chunks (one-time operation)
-- This will compress all chunks older than 7 days
SELECT compress_chunk(i, if_not_compressed => true)
FROM show_chunks('price_data_5min', older_than => INTERVAL '7 days') i;

-- Verify compression status
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end,
    is_compressed,
    pg_size_pretty(before_compression_total_bytes) as before_size,
    pg_size_pretty(after_compression_total_bytes) as after_size
FROM timescaledb_information.chunks
WHERE hypertable_name = 'price_data_5min'
ORDER BY range_start DESC
LIMIT 10;
