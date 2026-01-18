-- Migration: Experiments tracking table
-- Purpose: Track A/B experiments for model hyperparameters and training configurations

CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_id TEXT UNIQUE NOT NULL,  -- e.g., 'training_window_1mo_vs_6mo'
    name TEXT NOT NULL,
    description TEXT,

    -- Experiment configuration (stored as JSON)
    config JSONB NOT NULL DEFAULT '{}',

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),

    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results summary (populated on completion)
    results JSONB
);

CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created ON experiments(created_at DESC);

-- Experiment variants (each configuration being tested)
CREATE TABLE IF NOT EXISTS experiment_variants (
    id SERIAL PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    variant_name TEXT NOT NULL,  -- e.g., 'control_6mo', 'treatment_1mo'

    -- Variant-specific config overrides
    config JSONB NOT NULL DEFAULT '{}',

    -- Training run association
    run_id TEXT,  -- Links to training_jobs.run_id

    -- Results
    item_count INTEGER,
    mean_auc DECIMAL(6,4),
    median_auc DECIMAL(6,4),
    std_auc DECIMAL(6,4),
    min_auc DECIMAL(6,4),
    max_auc DECIMAL(6,4),

    -- Per-target breakdown (108 targets)
    target_aucs JSONB,  -- {"target_0": 0.65, "target_1": 0.62, ...}

    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    UNIQUE(experiment_id, variant_name)
);

CREATE INDEX idx_experiment_variants_experiment ON experiment_variants(experiment_id);
CREATE INDEX idx_experiment_variants_run ON experiment_variants(run_id);

-- Comments
COMMENT ON TABLE experiments IS 'Tracks A/B experiments for model training configurations';
COMMENT ON TABLE experiment_variants IS 'Individual variants within an experiment with their results';
COMMENT ON COLUMN experiments.config IS 'Base experiment configuration as JSON';
COMMENT ON COLUMN experiment_variants.config IS 'Variant-specific config overrides (e.g., months_history: 1)';
