-- Migration: 002_create_model_registry_table
-- Description: Creates the model_registry table for ML model lifecycle management
-- Issue: #51 - Add model lifecycle status handling
-- Date: 2026-01-11

-- Create model_registry table for tracking model lifecycle status
-- Note: This table may already exist if created manually. Using IF NOT EXISTS for safety.
CREATE TABLE IF NOT EXISTS model_registry (
    model_id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL,                    -- OSRS item ID this model predicts for
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE', -- ACTIVE, DEPRECATED, SUNSET, ARCHIVED
    mean_auc FLOAT,                              -- Model quality metric (area under ROC curve)
    trained_at TIMESTAMP WITH TIME ZONE,         -- When the model was trained
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_status CHECK (status IN ('ACTIVE', 'DEPRECATED', 'SUNSET', 'ARCHIVED'))
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_model_registry_item_id ON model_registry(item_id);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_item_status ON model_registry(item_id, status);

-- Comments for documentation
COMMENT ON TABLE model_registry IS 'Tracks ML model lifecycle status - ensures recommendations only use ACTIVE models while supporting existing trades on deprecated models';
COMMENT ON COLUMN model_registry.status IS 'Model lifecycle status: ACTIVE (use for new recommendations), DEPRECATED (no new trades), SUNSET (being phased out), ARCHIVED (no longer in use)';
COMMENT ON COLUMN model_registry.mean_auc IS 'Model quality metric - area under ROC curve from training evaluation';
