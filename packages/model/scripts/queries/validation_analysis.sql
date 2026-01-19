-- packages/model/scripts/queries/validation_analysis.sql
-- Useful queries for analyzing prediction validation data

-- ============================================================
-- QUERY 1: Overall accuracy summary (last 7 days)
-- ============================================================
SELECT
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS') as clear_misses,
    COUNT(*) FILTER (WHERE outcome = 'CLEAR_HIT') as clear_hits,
    COUNT(*) FILTER (WHERE outcome = 'POSSIBLE_HIT') as possible_hits,
    COUNT(*) FILTER (WHERE outcome = 'POSSIBLE_MISS') as possible_misses,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as clear_miss_rate_pct
FROM prediction_outcomes
WHERE created_at > NOW() - INTERVAL '7 days';


-- ============================================================
-- QUERY 2: Miss rate by item (worst performers)
-- ============================================================
SELECT
    po.item_id,
    i.name as item_name,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE po.outcome = 'CLEAR_MISS') as clear_misses,
    ROUND(
        COUNT(*) FILTER (WHERE po.outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as miss_rate_pct
FROM prediction_outcomes po
LEFT JOIN items i ON i.id = po.item_id
WHERE po.created_at > NOW() - INTERVAL '7 days'
GROUP BY po.item_id, i.name
HAVING COUNT(*) >= 10  -- Minimum sample size
ORDER BY miss_rate_pct DESC
LIMIT 20;


-- ============================================================
-- QUERY 3: Miss rate by model version
-- ============================================================
SELECT
    model_version,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS') as clear_misses,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as miss_rate_pct
FROM prediction_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
  AND model_version IS NOT NULL
GROUP BY model_version
ORDER BY miss_rate_pct DESC;


-- ============================================================
-- QUERY 4: Miss rate by hour_offset (which prediction windows work best)
-- ============================================================
SELECT
    hour_offset,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS') as clear_misses,
    COUNT(*) FILTER (WHERE outcome = 'POSSIBLE_HIT') as possible_hits,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as miss_rate_pct
FROM prediction_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY hour_offset
ORDER BY hour_offset;


-- ============================================================
-- QUERY 5: Calibration check (are 80% predictions ~80% accurate?)
-- ============================================================
SELECT
    CASE
        WHEN predicted_fill_probability < 0.2 THEN '0-20%'
        WHEN predicted_fill_probability < 0.4 THEN '20-40%'
        WHEN predicted_fill_probability < 0.6 THEN '40-60%'
        WHEN predicted_fill_probability < 0.8 THEN '60-80%'
        ELSE '80-100%'
    END as confidence_bucket,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE outcome IN ('POSSIBLE_HIT', 'CLEAR_HIT')) as hits,
    ROUND(
        COUNT(*) FILTER (WHERE outcome IN ('POSSIBLE_HIT', 'CLEAR_HIT'))::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as hit_rate_pct
FROM prediction_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- QUERY 6: Daily trend (is accuracy improving or degrading?)
-- ============================================================
SELECT
    DATE_TRUNC('day', prediction_time) as prediction_date,
    COUNT(*) as total_predictions,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as miss_rate_pct
FROM prediction_outcomes
WHERE created_at > NOW() - INTERVAL '14 days'
GROUP BY DATE_TRUNC('day', prediction_time)
ORDER BY prediction_date DESC;


-- ============================================================
-- QUERY 7: Items needing retraining (high miss rate + stale model)
-- ============================================================
SELECT
    po.item_id,
    i.name as item_name,
    po.model_version,
    COUNT(*) as predictions_evaluated,
    ROUND(
        COUNT(*) FILTER (WHERE po.outcome = 'CLEAR_MISS')::DECIMAL /
        NULLIF(COUNT(*), 0) * 100,
        2
    ) as miss_rate_pct,
    mr.created_at as model_created_at,
    EXTRACT(DAY FROM NOW() - mr.created_at) as model_age_days
FROM prediction_outcomes po
LEFT JOIN items i ON i.id = po.item_id
LEFT JOIN model_registry mr ON mr.item_id = po.item_id AND mr.status = 'ACTIVE'
WHERE po.created_at > NOW() - INTERVAL '7 days'
GROUP BY po.item_id, i.name, po.model_version, mr.created_at
HAVING COUNT(*) >= 10
   AND (
       -- High miss rate
       COUNT(*) FILTER (WHERE po.outcome = 'CLEAR_MISS')::DECIMAL /
       NULLIF(COUNT(*), 0) > 0.3
       -- OR stale model
       OR EXTRACT(DAY FROM NOW() - mr.created_at) > 14
   )
ORDER BY miss_rate_pct DESC;
