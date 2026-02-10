-- Migration 002: Training Jobs Table
-- Tracks training pipeline runs and their status
--
-- Run: psql -h localhost -U osrs_user -d osrs_data -f scripts/migrations/002_training_jobs.sql

BEGIN;

CREATE TABLE IF NOT EXISTS training_jobs (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,             -- e.g., '20260111_050000'

    -- Job status
    status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING, SELECTING, PREPARING, TRAINING, VALIDATING, DEPLOYING, COMPLETED, FAILED
    status_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Selection phase
    selection_criteria JSONB,                -- {min_volume, min_rows, max_age_days, ...}
    items_selected JSONB,                    -- [{item_id, item_name, reason}, ...]
    items_count INTEGER DEFAULT 0,

    -- Training phase results
    items_trained INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    training_errors JSONB,                   -- [{item_id, error}, ...]

    -- Validation phase results
    items_validated INTEGER DEFAULT 0,
    items_validation_passed INTEGER DEFAULT 0,
    items_validation_failed INTEGER DEFAULT 0,

    -- Deployment phase results
    items_deployed INTEGER DEFAULT 0,
    items_skipped INTEGER DEFAULT 0,         -- Already have better model

    -- Error handling
    error_message TEXT,
    error_traceback TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Execution environment
    training_host TEXT,                      -- WSL hostname/IP
    gpu_info TEXT,                           -- 'RTX 3060 12GB'
    training_duration_seconds REAL,          -- Total training time

    -- Trigger info
    trigger_type TEXT DEFAULT 'scheduled',   -- scheduled, manual, discovery
    triggered_by TEXT,                       -- 'cron', 'user:marcus', etc.

    CHECK (status IN ('PENDING', 'SELECTING', 'PREPARING', 'TRAINING', 'VALIDATING', 'DEPLOYING', 'COMPLETED', 'FAILED'))
);

-- Comments
COMMENT ON TABLE training_jobs IS 'Training pipeline execution history';
COMMENT ON COLUMN training_jobs.trigger_type IS 'scheduled=daily cron, manual=user triggered, discovery=monthly full scan';
COMMENT ON COLUMN training_jobs.items_selected IS 'JSON array of {item_id, item_name, reason} for items chosen for training';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_training_jobs_status
    ON training_jobs(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_jobs_created
    ON training_jobs(created_at DESC);

-- View: Recent training jobs
CREATE OR REPLACE VIEW recent_training_jobs AS
SELECT
    tj.id,
    tj.run_id,
    tj.status,
    tj.trigger_type,
    tj.items_count,
    tj.items_trained,
    tj.items_deployed,
    tj.items_failed,
    tj.training_host,
    tj.created_at,
    tj.completed_at,
    CASE
        WHEN tj.completed_at IS NOT NULL THEN
            EXTRACT(EPOCH FROM tj.completed_at - tj.started_at) / 60
        WHEN tj.started_at IS NOT NULL THEN
            EXTRACT(EPOCH FROM NOW() - tj.started_at) / 60
        ELSE NULL
    END AS duration_minutes,
    tj.error_message
FROM training_jobs tj
ORDER BY tj.created_at DESC
LIMIT 50;

COMMENT ON VIEW recent_training_jobs IS 'Last 50 training jobs with duration';

-- View: Training job success rate
CREATE OR REPLACE VIEW training_job_stats AS
SELECT
    DATE_TRUNC('day', created_at) AS job_date,
    COUNT(*) AS total_jobs,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') AS successful,
    COUNT(*) FILTER (WHERE status = 'FAILED') AS failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'COMPLETED') / NULLIF(COUNT(*), 0), 1) AS success_rate_pct,
    SUM(items_deployed) AS total_models_deployed,
    ROUND(AVG(training_duration_seconds)::numeric / 60, 1) AS avg_duration_minutes
FROM training_jobs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY job_date DESC;

COMMENT ON VIEW training_job_stats IS 'Daily training job statistics for last 30 days';

COMMIT;
