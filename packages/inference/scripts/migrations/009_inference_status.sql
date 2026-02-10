-- Migration 009: Inference Status Table
-- Track inference run status for freshness verification
-- Allows downstream systems to verify predictions are current

-- Inference status tracking
CREATE TABLE IF NOT EXISTS inference_status (
    id SERIAL PRIMARY KEY,
    inference_started_at TIMESTAMPTZ NOT NULL,
    inference_completed_at TIMESTAMPTZ,
    run_id TEXT NOT NULL,
    model_run_id TEXT NOT NULL,
    model_trained_at TIMESTAMPTZ,
    items_predicted INTEGER DEFAULT 0,
    predictions_written INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for quick "latest successful" lookup
CREATE INDEX IF NOT EXISTS idx_inference_status_latest
ON inference_status (inference_completed_at DESC)
WHERE status = 'completed';

-- Index for cleanup queries
CREATE INDEX IF NOT EXISTS idx_inference_status_created
ON inference_status (created_at);

COMMENT ON TABLE inference_status IS 'Tracks inference run status for freshness verification';
COMMENT ON COLUMN inference_status.run_id IS 'Unique identifier for this inference run';
COMMENT ON COLUMN inference_status.model_run_id IS 'Model training run ID, e.g., 20260112_032324';
COMMENT ON COLUMN inference_status.status IS 'running, completed, or failed';
