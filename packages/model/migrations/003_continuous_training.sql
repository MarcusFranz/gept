-- Migration 003: Continuous Training Pipeline Schema
-- Date: 2026-01-17
-- Description: Add tables for continuous training with drift-based prioritization

-- Item-specific optimized hyperparameters
CREATE TABLE IF NOT EXISTS item_hyperparameters (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    category VARCHAR(50) NOT NULL DEFAULT 'default',
    depth INTEGER NOT NULL DEFAULT 4,
    learning_rate REAL NOT NULL DEFAULT 0.1,
    l2_leaf_reg REAL NOT NULL DEFAULT 3.0,
    days_history INTEGER NOT NULL DEFAULT 60,
    optimized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    optimization_auc REAL,
    optimization_method VARCHAR(50) DEFAULT 'category_default',
    notes TEXT
);

-- Create index for category lookups
CREATE INDEX IF NOT EXISTS idx_item_hyperparameters_category
    ON item_hyperparameters(category);

-- Training job history (audit trail)
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    job_type VARCHAR(30) NOT NULL,  -- 'TRAIN', 'OPTIMIZE', 'OPTIMIZE_AND_TRAIN'
    priority INTEGER NOT NULL,
    drift_severity VARCHAR(20),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'RUNNING',  -- 'RUNNING', 'COMPLETED', 'FAILED'
    result_auc REAL,
    training_time_seconds REAL,
    error_message TEXT,
    run_id VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for training_jobs
CREATE INDEX IF NOT EXISTS idx_training_jobs_item_id ON training_jobs(item_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_started_at ON training_jobs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_run_id ON training_jobs(run_id);

-- Add drift columns to model_performance if they don't exist
DO $$
BEGIN
    -- baseline_auc column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'model_performance' AND column_name = 'baseline_auc'
    ) THEN
        ALTER TABLE model_performance ADD COLUMN baseline_auc REAL;
    END IF;

    -- auc_trend column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'model_performance' AND column_name = 'auc_trend'
    ) THEN
        ALTER TABLE model_performance ADD COLUMN auc_trend REAL;
    END IF;

    -- drift_severity column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'model_performance' AND column_name = 'drift_severity'
    ) THEN
        ALTER TABLE model_performance ADD COLUMN drift_severity VARCHAR(20);
    END IF;
END $$;

-- Create index for drift detection queries
CREATE INDEX IF NOT EXISTS idx_model_performance_drift
    ON model_performance(item_id, drift_severity)
    WHERE drift_severity IS NOT NULL;

-- View: Items with critical or warning drift
CREATE OR REPLACE VIEW v_items_with_drift AS
WITH performance_stats AS (
    SELECT
        item_id,
        AVG(estimated_auc) FILTER (WHERE time > NOW() - INTERVAL '7 days') as baseline_auc,
        AVG(estimated_auc) FILTER (WHERE time > NOW() - INTERVAL '24 hours') as current_auc,
        regr_slope(estimated_auc, EXTRACT(EPOCH FROM time))
            FILTER (WHERE time > NOW() - INTERVAL '7 days') as auc_trend
    FROM model_performance
    WHERE window_hours = 24
      AND estimated_auc IS NOT NULL
    GROUP BY item_id
    HAVING COUNT(*) FILTER (WHERE time > NOW() - INTERVAL '7 days') >= 5
)
SELECT
    ps.item_id,
    i.name as item_name,
    ps.baseline_auc,
    ps.current_auc,
    ps.auc_trend,
    (ps.baseline_auc - ps.current_auc) / NULLIF(ps.baseline_auc, 0) as auc_drop_pct,
    CASE
        WHEN (ps.baseline_auc - ps.current_auc) / NULLIF(ps.baseline_auc, 0) > 0.10 THEN 'CRITICAL'
        WHEN (ps.baseline_auc - ps.current_auc) / NULLIF(ps.baseline_auc, 0) > 0.05 THEN 'WARNING'
        WHEN ps.auc_trend < -0.01 THEN 'PREDICTED'
        ELSE 'STABLE'
    END as drift_severity
FROM performance_stats ps
LEFT JOIN items i ON i.item_id = ps.item_id
ORDER BY
    CASE
        WHEN (ps.baseline_auc - ps.current_auc) / NULLIF(ps.baseline_auc, 0) > 0.10 THEN 0
        WHEN (ps.baseline_auc - ps.current_auc) / NULLIF(ps.baseline_auc, 0) > 0.05 THEN 1
        WHEN ps.auc_trend < -0.01 THEN 2
        ELSE 3
    END,
    (ps.baseline_auc - ps.current_auc) / NULLIF(ps.baseline_auc, 0) DESC;

-- View: Training pipeline summary
CREATE OR REPLACE VIEW v_training_pipeline_summary AS
SELECT
    DATE_TRUNC('day', started_at) as day,
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
    COUNT(*) FILTER (WHERE status = 'FAILED') as failed,
    COUNT(*) FILTER (WHERE status = 'RUNNING') as running,
    AVG(training_time_seconds) FILTER (WHERE status = 'COMPLETED') as avg_training_time,
    AVG(result_auc) FILTER (WHERE status = 'COMPLETED') as avg_auc,
    COUNT(DISTINCT item_id) as unique_items
FROM training_jobs
WHERE started_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', started_at)
ORDER BY day DESC;

-- Function: Get training priority queue
CREATE OR REPLACE FUNCTION get_training_priority_queue(p_limit INTEGER DEFAULT 50)
RETURNS TABLE (
    item_id INTEGER,
    item_name VARCHAR,
    model_id INTEGER,
    model_age_days NUMERIC,
    drift_severity VARCHAR,
    auc_drop_pct NUMERIC,
    priority INTEGER,
    recommended_action VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    WITH item_data AS (
        SELECT DISTINCT p.item_id
        FROM price_data_5min p
        WHERE p.timestamp > NOW() - INTERVAL '60 days'
        GROUP BY p.item_id
        HAVING COUNT(*) >= 5000
    ),
    active_models AS (
        SELECT
            mr.item_id,
            mr.id as model_id,
            EXTRACT(EPOCH FROM NOW() - mr.trained_at) / 86400 as model_age_days
        FROM model_registry mr
        WHERE mr.status = 'ACTIVE'
    ),
    drift_info AS (
        SELECT
            d.item_id,
            d.drift_severity,
            d.auc_drop_pct::NUMERIC
        FROM v_items_with_drift d
    )
    SELECT
        id.item_id,
        i.name::VARCHAR as item_name,
        am.model_id::INTEGER,
        COALESCE(am.model_age_days, 999)::NUMERIC as model_age_days,
        COALESCE(di.drift_severity, 'STABLE')::VARCHAR as drift_severity,
        COALESCE(di.auc_drop_pct, 0)::NUMERIC as auc_drop_pct,
        CASE
            WHEN am.model_id IS NULL THEN 5  -- No model
            WHEN di.drift_severity = 'CRITICAL' THEN 0
            WHEN di.drift_severity = 'WARNING' THEN 1
            WHEN di.drift_severity = 'PREDICTED' THEN 2
            WHEN am.model_age_days > 7 THEN 3
            ELSE 4
        END::INTEGER as priority,
        CASE
            WHEN am.model_id IS NULL THEN 'TRAIN'
            WHEN di.drift_severity = 'CRITICAL' THEN 'OPTIMIZE_AND_TRAIN'
            WHEN di.drift_severity IN ('WARNING', 'PREDICTED') THEN 'TRAIN'
            WHEN am.model_age_days > 7 THEN 'TRAIN'
            ELSE 'OPTIMIZE'
        END::VARCHAR as recommended_action
    FROM item_data id
    LEFT JOIN items i ON i.item_id = id.item_id
    LEFT JOIN active_models am ON am.item_id = id.item_id
    LEFT JOIN drift_info di ON di.item_id = id.item_id
    ORDER BY priority, auc_drop_pct DESC NULLS LAST, model_age_days DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT SELECT ON v_items_with_drift TO osrs_user;
GRANT SELECT ON v_training_pipeline_summary TO osrs_user;
GRANT EXECUTE ON FUNCTION get_training_priority_queue(INTEGER) TO osrs_user;

-- Insert category defaults into item_hyperparameters for known items
-- High-volume consumables
INSERT INTO item_hyperparameters (item_id, category, depth, learning_rate, l2_leaf_reg, days_history, optimization_method)
SELECT i.item_id, 'high_volume_consumables', 3, 0.15, 1.0, 60, 'experiment_results'
FROM items i
WHERE i.item_id IN (2, 561, 560, 562, 563, 564, 565, 566, 554, 555, 556, 557, 558, 559, 9075, 892, 11212, 11230, 21905)
ON CONFLICT (item_id) DO UPDATE SET
    category = EXCLUDED.category,
    depth = EXCLUDED.depth,
    learning_rate = EXCLUDED.learning_rate,
    l2_leaf_reg = EXCLUDED.l2_leaf_reg;

-- Mid-volume resources
INSERT INTO item_hyperparameters (item_id, category, depth, learning_rate, l2_leaf_reg, days_history, optimization_method)
SELECT i.item_id, 'mid_volume_resources', 5, 0.15, 10.0, 60, 'experiment_results'
FROM items i
WHERE i.item_id IN (1515, 1513, 1517, 453, 440, 444, 447, 449, 451, 2351, 2353, 2357, 2359, 2361, 2363, 1779, 5295, 5300)
ON CONFLICT (item_id) DO UPDATE SET
    category = EXCLUDED.category,
    depth = EXCLUDED.depth,
    learning_rate = EXCLUDED.learning_rate,
    l2_leaf_reg = EXCLUDED.l2_leaf_reg;

-- Potions and food
INSERT INTO item_hyperparameters (item_id, category, depth, learning_rate, l2_leaf_reg, days_history, optimization_method)
SELECT i.item_id, 'potions_food', 6, 0.05, 5.0, 60, 'experiment_results'
FROM items i
WHERE i.item_id IN (3024, 2434, 2442, 12695, 3040, 2444, 6685, 385, 7946, 391, 3144, 11936, 13441)
ON CONFLICT (item_id) DO UPDATE SET
    category = EXCLUDED.category,
    depth = EXCLUDED.depth,
    learning_rate = EXCLUDED.learning_rate,
    l2_leaf_reg = EXCLUDED.l2_leaf_reg;

-- High-value equipment (longer history)
INSERT INTO item_hyperparameters (item_id, category, depth, learning_rate, l2_leaf_reg, days_history, optimization_method)
SELECT i.item_id, 'equipment_high_value', 4, 0.1, 3.0, 180, 'experiment_results'
FROM items i
WHERE i.item_id IN (11802, 11804, 11806, 11808, 11832, 11834, 11826, 11828, 11830, 12817, 21034, 22325, 22324, 13576, 12002, 12924, 21018, 21021, 21024)
ON CONFLICT (item_id) DO UPDATE SET
    category = EXCLUDED.category,
    days_history = EXCLUDED.days_history;

COMMENT ON TABLE item_hyperparameters IS 'Item-specific hyperparameters from optimization experiments';
COMMENT ON TABLE training_jobs IS 'Audit trail of all training jobs executed by the continuous scheduler';
COMMENT ON VIEW v_items_with_drift IS 'Items with detected model drift requiring retraining';
COMMENT ON VIEW v_training_pipeline_summary IS 'Daily summary of training pipeline activity';
