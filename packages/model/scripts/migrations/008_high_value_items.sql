-- Migration 008: High-Value Item Metadata (Issue #120)
-- Tracks assessment results for high-value items to support training decisions
-- and manipulation risk monitoring.

-- High-value item metadata table
CREATE TABLE IF NOT EXISTS high_value_item_metadata (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
    assessed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Data quality metrics
    data_quality_score DECIMAL(4,3) CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    completeness DECIMAL(5,4) CHECK (completeness >= 0 AND completeness <= 1),
    history_days DECIMAL(6,1),
    data_rows INTEGER,

    -- Manipulation risk assessment
    manipulation_risk TEXT CHECK (manipulation_risk IN ('low', 'medium', 'high')),
    volume_stability DECIMAL(4,3) CHECK (volume_stability >= 0 AND volume_stability <= 1),
    spread_stability DECIMAL(4,3) CHECK (spread_stability >= 0 AND spread_stability <= 1),
    volume_spike_count INTEGER DEFAULT 0,

    -- Training viability
    training_viable BOOLEAN DEFAULT false,
    viability_reason TEXT,

    -- Warnings as JSON array
    warnings JSONB DEFAULT '[]'::jsonb,

    -- Tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for quick lookups of viable items
CREATE INDEX IF NOT EXISTS idx_hv_viable_score
ON high_value_item_metadata(training_viable, data_quality_score DESC)
WHERE training_viable = true;

-- Index for manipulation risk monitoring
CREATE INDEX IF NOT EXISTS idx_hv_manipulation_risk
ON high_value_item_metadata(manipulation_risk, assessed_at DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_hv_metadata_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_hv_metadata_timestamp ON high_value_item_metadata;
CREATE TRIGGER update_hv_metadata_timestamp
BEFORE UPDATE ON high_value_item_metadata
FOR EACH ROW
EXECUTE FUNCTION update_hv_metadata_timestamp();

-- View for high-value items ready for training
CREATE OR REPLACE VIEW high_value_items_ready AS
SELECT
    hv.item_id,
    i.name as item_name,
    i.value as item_price,
    hv.data_quality_score,
    hv.manipulation_risk,
    hv.training_viable,
    hv.viability_reason,
    hv.assessed_at,
    -- Check if model exists
    EXISTS (
        SELECT 1 FROM model_registry mr
        WHERE mr.item_id = hv.item_id AND mr.status = 'ACTIVE'
    ) as has_model
FROM high_value_item_metadata hv
JOIN items i ON i.item_id = hv.item_id
WHERE hv.training_viable = true
ORDER BY i.value DESC;

-- View for high-value items needing attention (high risk or no model)
CREATE OR REPLACE VIEW high_value_items_attention AS
SELECT
    hv.item_id,
    i.name as item_name,
    i.value as item_price,
    hv.manipulation_risk,
    hv.training_viable,
    hv.viability_reason,
    hv.warnings,
    hv.assessed_at,
    -- Check if model exists
    NOT EXISTS (
        SELECT 1 FROM model_registry mr
        WHERE mr.item_id = hv.item_id AND mr.status = 'ACTIVE'
    ) as needs_model
FROM high_value_item_metadata hv
JOIN items i ON i.item_id = hv.item_id
WHERE hv.manipulation_risk = 'high'
   OR (hv.training_viable = true AND NOT EXISTS (
        SELECT 1 FROM model_registry mr
        WHERE mr.item_id = hv.item_id AND mr.status = 'ACTIVE'
   ))
ORDER BY i.value DESC;

-- Add comment
COMMENT ON TABLE high_value_item_metadata IS
'Stores assessment results for high-value items (Issue #120). Used to track data quality, manipulation risk, and training viability for items >= 10M gp.';
