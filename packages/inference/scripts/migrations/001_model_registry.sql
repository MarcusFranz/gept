-- Migration 001: Model Registry Table
-- Tracks all trained models and their lifecycle states
--
-- Run: psql -h localhost -U osrs_user -d osrs_data -f scripts/migrations/001_model_registry.sql

BEGIN;

-- Create enum type for model status (if not using CHECK constraint)
-- Using CHECK constraint for broader compatibility

CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    run_id TEXT NOT NULL,                    -- e.g., '20260111_050000'
    model_path TEXT NOT NULL,                -- Relative path: models/<run_id>/<item_id>/model.cbm

    -- Status lifecycle
    status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING, ACTIVE, DEPRECATED, SUNSET, ARCHIVED
    status_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status_reason TEXT,                      -- e.g., 'AUC degradation detected', 'Replaced by newer model'

    -- Training metadata
    trained_at TIMESTAMPTZ NOT NULL,
    training_duration_seconds REAL,

    -- Model metrics (from training)
    mean_auc DECIMAL(5,4),                   -- Average AUC across all 108 targets
    targets_above_52 INTEGER,                -- Count of targets with AUC > 0.52
    targets_above_55 INTEGER,                -- Count of targets with AUC > 0.55
    targets_scored INTEGER,                  -- Total targets that could be scored
    n_features INTEGER,                      -- Number of features used
    n_samples INTEGER,                       -- Training samples count
    iterations_used INTEGER,                 -- Training iterations (may be less than max due to early stopping)

    -- Validation metrics (from deployment check)
    validation_auc DECIMAL(5,4),             -- AUC on validation set
    validation_passed BOOLEAN,
    validated_at TIMESTAMPTZ,
    validation_notes TEXT,

    -- Lifecycle tracking timestamps
    activated_at TIMESTAMPTZ,                -- When model went ACTIVE
    deprecated_at TIMESTAMPTZ,               -- When model was replaced
    sunset_at TIMESTAMPTZ,                   -- When 48h grace period started
    archived_at TIMESTAMPTZ,                 -- When model was archived

    -- Model lineage tracking
    replaces_model_id BIGINT REFERENCES model_registry(id),
    replaced_by_model_id BIGINT REFERENCES model_registry(id),

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE (item_id, run_id),
    CHECK (status IN ('PENDING', 'ACTIVE', 'DEPRECATED', 'SUNSET', 'ARCHIVED'))
);

-- Comments
COMMENT ON TABLE model_registry IS 'Registry of all trained models with lifecycle tracking';
COMMENT ON COLUMN model_registry.status IS 'PENDING=awaiting validation, ACTIVE=in production, DEPRECATED=replaced, SUNSET=grace period, ARCHIVED=removed';
COMMENT ON COLUMN model_registry.mean_auc IS 'Average AUC across all 108 targets (18 windows x 6 offsets)';
COMMENT ON COLUMN model_registry.sunset_at IS 'Start of 48-hour grace period before archival';

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_model_registry_item_active
    ON model_registry(item_id)
    WHERE status = 'ACTIVE';

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON model_registry(status, status_changed_at);

CREATE INDEX IF NOT EXISTS idx_model_registry_run
    ON model_registry(run_id);

CREATE INDEX IF NOT EXISTS idx_model_registry_item_status
    ON model_registry(item_id, status);

CREATE INDEX IF NOT EXISTS idx_model_registry_trained_at
    ON model_registry(trained_at DESC);

-- View: Current active models
CREATE OR REPLACE VIEW active_models AS
SELECT
    mr.id,
    mr.item_id,
    mr.item_name,
    mr.run_id,
    mr.model_path,
    mr.mean_auc,
    mr.targets_above_55,
    mr.trained_at,
    mr.activated_at,
    EXTRACT(EPOCH FROM NOW() - mr.trained_at) / 86400 AS model_age_days
FROM model_registry mr
WHERE mr.status = 'ACTIVE'
ORDER BY mr.item_id;

COMMENT ON VIEW active_models IS 'Currently active production models';

-- View: Model status summary
CREATE OR REPLACE VIEW model_status_summary AS
SELECT
    status,
    COUNT(*) as model_count,
    MIN(trained_at) as oldest_trained,
    MAX(trained_at) as newest_trained,
    ROUND(AVG(mean_auc)::numeric, 4) as avg_auc,
    ROUND(AVG(targets_above_55)::numeric, 1) as avg_targets_above_55
FROM model_registry
GROUP BY status
ORDER BY
    CASE status
        WHEN 'ACTIVE' THEN 1
        WHEN 'DEPRECATED' THEN 2
        WHEN 'SUNSET' THEN 3
        WHEN 'PENDING' THEN 4
        WHEN 'ARCHIVED' THEN 5
    END;

COMMENT ON VIEW model_status_summary IS 'Aggregate statistics by model status';

-- View: Models needing attention (stale, degraded, or in sunset)
CREATE OR REPLACE VIEW models_needing_attention AS
SELECT
    mr.id,
    mr.item_id,
    mr.item_name,
    mr.status,
    mr.mean_auc,
    mr.trained_at,
    EXTRACT(EPOCH FROM NOW() - mr.trained_at) / 86400 AS model_age_days,
    CASE
        WHEN mr.status = 'SUNSET' THEN
            EXTRACT(EPOCH FROM (mr.sunset_at + INTERVAL '48 hours') - NOW()) / 3600
        ELSE NULL
    END AS hours_until_archive,
    mr.status_reason
FROM model_registry mr
WHERE
    mr.status = 'SUNSET'
    OR (mr.status = 'ACTIVE' AND mr.trained_at < NOW() - INTERVAL '30 days')
ORDER BY
    CASE mr.status WHEN 'SUNSET' THEN 1 ELSE 2 END,
    mr.trained_at ASC;

COMMENT ON VIEW models_needing_attention IS 'Models in sunset or stale active models needing retraining';

COMMIT;
