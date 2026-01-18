-- =============================================================================
-- Migration 007: Data Quality Tracking Table
-- =============================================================================
-- Tracks data collection gaps and backfill status for defense in depth monitoring
-- =============================================================================

BEGIN;

-- Data quality tracking table
CREATE TABLE IF NOT EXISTS data_quality (
    id BIGSERIAL PRIMARY KEY,

    -- Gap identification
    table_name TEXT NOT NULL,
    item_id INTEGER,                    -- NULL for table-level gaps
    gap_start TIMESTAMPTZ NOT NULL,
    gap_end TIMESTAMPTZ NOT NULL,
    gap_duration_seconds INTEGER GENERATED ALWAYS AS
        (EXTRACT(EPOCH FROM (gap_end - gap_start))::INTEGER) STORED,

    -- Gap status
    status TEXT NOT NULL DEFAULT 'DETECTED',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,

    -- Backfill metadata
    backfill_attempts INTEGER DEFAULT 0,
    backfill_source TEXT,               -- 'wiki_api', 'manual', etc.
    rows_recovered INTEGER DEFAULT 0,

    -- Error tracking
    error_message TEXT,

    -- Constraints
    CHECK (status IN ('DETECTED', 'BACKFILLING', 'BACKFILLED', 'UNRECOVERABLE', 'IGNORED')),
    UNIQUE (table_name, item_id, gap_start)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_data_quality_status
    ON data_quality(status)
    WHERE status NOT IN ('BACKFILLED', 'IGNORED');

CREATE INDEX IF NOT EXISTS idx_data_quality_table
    ON data_quality(table_name, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_data_quality_unresolved
    ON data_quality(table_name, gap_start)
    WHERE status = 'DETECTED';

CREATE INDEX IF NOT EXISTS idx_data_quality_item
    ON data_quality(item_id, gap_start)
    WHERE item_id IS NOT NULL;

-- Summary view for monitoring dashboards
CREATE OR REPLACE VIEW data_quality_summary AS
SELECT
    table_name,
    status,
    COUNT(*) as gap_count,
    SUM(gap_duration_seconds) as total_gap_seconds,
    MIN(gap_start) as earliest_gap,
    MAX(gap_end) as latest_gap,
    SUM(rows_recovered) as total_rows_recovered
FROM data_quality
GROUP BY table_name, status
ORDER BY table_name, status;

-- Active gaps view for alerting
CREATE OR REPLACE VIEW active_gaps AS
SELECT
    id,
    table_name,
    item_id,
    gap_start,
    gap_end,
    gap_duration_seconds,
    status,
    detected_at,
    backfill_attempts,
    error_message
FROM data_quality
WHERE status IN ('DETECTED', 'BACKFILLING')
ORDER BY gap_start DESC;

-- Recent gaps view for debugging
CREATE OR REPLACE VIEW recent_gaps AS
SELECT
    id,
    table_name,
    item_id,
    gap_start,
    gap_end,
    gap_duration_seconds,
    status,
    detected_at,
    resolved_at,
    backfill_source,
    rows_recovered
FROM data_quality
WHERE detected_at > NOW() - INTERVAL '7 days'
ORDER BY detected_at DESC;

-- Add comments for documentation
COMMENT ON TABLE data_quality IS 'Tracks data collection gaps and backfill status for defense in depth monitoring';
COMMENT ON COLUMN data_quality.table_name IS 'Source table where gap was detected (e.g., price_data_5min)';
COMMENT ON COLUMN data_quality.item_id IS 'Specific item ID if gap is item-specific, NULL for table-wide gaps';
COMMENT ON COLUMN data_quality.gap_start IS 'Start timestamp of the data gap';
COMMENT ON COLUMN data_quality.gap_end IS 'End timestamp of the data gap';
COMMENT ON COLUMN data_quality.status IS 'Current status: DETECTED, BACKFILLING, BACKFILLED, UNRECOVERABLE, IGNORED';
COMMENT ON COLUMN data_quality.backfill_source IS 'Source used for backfill: wiki_api, manual, etc.';

COMMIT;
