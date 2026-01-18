-- Migration: Add retention policy for prices_latest_1m
-- Issue: #147
-- Date: 2026-01-18
--
-- This migration adds a TimescaleDB retention policy to automatically
-- delete old 1-minute price data. The 1-minute data is primarily used
-- for real-time inference; historical analysis uses 5-minute data.
--
-- Current state:
--   - 63M rows, 10GB storage
--   - No retention policy
--   - Data accumulates indefinitely
--
-- After migration:
--   - Data older than 30 days automatically deleted
--   - Expected size reduction: ~7-8GB (keeping only last 30 days)

-- Check if the table is a hypertable (required for retention policy)
-- If not, you'll need to convert it first with:
-- SELECT create_hypertable('prices_latest_1m', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);

-- Add retention policy to delete data older than 30 days
-- This runs automatically via TimescaleDB background workers
SELECT add_retention_policy(
    'prices_latest_1m',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Verify the policy was created
SELECT * FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention'
  AND hypertable_name = 'prices_latest_1m';

-- To manually run retention (useful for initial cleanup):
-- SELECT drop_chunks('prices_latest_1m', INTERVAL '30 days');

-- To check current chunk sizes:
-- SELECT hypertable_name, chunk_name, range_start, range_end,
--        pg_size_pretty(total_bytes) as size
-- FROM timescaledb_information.chunks
-- WHERE hypertable_name = 'prices_latest_1m'
-- ORDER BY range_start DESC;

-- To remove the policy later if needed:
-- SELECT remove_retention_policy('prices_latest_1m');
