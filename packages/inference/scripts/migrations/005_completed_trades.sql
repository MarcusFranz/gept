-- Migration 005: Completed Trades Table
-- Tracks completed trades for Discord bot history, portfolio, and ML feedback
--
-- Run: psql -h localhost -U osrs_user -d osrs_data -f scripts/migrations/005_completed_trades.sql

BEGIN;

CREATE TABLE IF NOT EXISTS completed_trades (
    id BIGSERIAL PRIMARY KEY,

    -- User identification (hashed for privacy)
    user_id_hash TEXT NOT NULL,

    -- Item reference
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,

    -- Trade details
    buy_price DECIMAL(12,2) NOT NULL,
    sell_price DECIMAL(12,2) NOT NULL,
    quantity INTEGER NOT NULL,

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'OPEN',

    -- Model/prediction reference (no FK due to TimescaleDB limitations)
    model_id BIGINT,                           -- References model_registry.id
    prediction_id TEXT,                        -- Composite key: {time}_{item_id}_{hour_offset}_{offset_pct}

    -- Prediction metadata (snapshot at trade creation)
    hour_offset INTEGER,
    offset_pct DECIMAL(5,4),
    predicted_fill_probability DECIMAL(7,6),

    -- Outcome tracking
    pnl DECIMAL(14,2),                         -- Realized profit/loss
    actual_filled BOOLEAN,                     -- Did both orders fill?

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    buy_filled_at TIMESTAMPTZ,
    sell_filled_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,

    -- Constraints
    CHECK (status IN ('OPEN', 'PARTIAL', 'COMPLETED', 'CANCELLED', 'EXPIRED')),
    CHECK (quantity > 0),
    CHECK (buy_price > 0),
    CHECK (sell_price > buy_price)
);

-- Comments
COMMENT ON TABLE completed_trades IS 'Tracks completed trades for Discord bot history, portfolio, and ML feedback exports';
COMMENT ON COLUMN completed_trades.user_id_hash IS 'SHA-256 hash of Discord user ID for privacy';
COMMENT ON COLUMN completed_trades.status IS 'OPEN=waiting for fills, PARTIAL=one side filled, COMPLETED=both filled, CANCELLED=user cancelled, EXPIRED=timed out';
COMMENT ON COLUMN completed_trades.model_id IS 'Reference to model_registry.id that generated the prediction';
COMMENT ON COLUMN completed_trades.prediction_id IS 'Composite key {time}_{item_id}_{hour_offset}_{offset_pct} referencing predictions table';
COMMENT ON COLUMN completed_trades.predicted_fill_probability IS 'Model fill probability snapshot at trade creation';
COMMENT ON COLUMN completed_trades.pnl IS 'Realized profit/loss: (sell_price - buy_price) * quantity - GE tax';
COMMENT ON COLUMN completed_trades.actual_filled IS 'Whether both buy and sell orders actually filled';

-- Indexes for common query patterns (from issue requirements)
CREATE INDEX IF NOT EXISTS idx_completed_trades_user_created
    ON completed_trades(user_id_hash, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_completed_trades_item_created
    ON completed_trades(item_id, created_at DESC);

-- Status filtering for active trades
CREATE INDEX IF NOT EXISTS idx_completed_trades_status
    ON completed_trades(status)
    WHERE status IN ('OPEN', 'PARTIAL');

-- ML feedback export queries
CREATE INDEX IF NOT EXISTS idx_completed_trades_model
    ON completed_trades(model_id)
    WHERE model_id IS NOT NULL;

-- Prediction tracking for calibration
CREATE INDEX IF NOT EXISTS idx_completed_trades_prediction
    ON completed_trades(prediction_id)
    WHERE prediction_id IS NOT NULL;

-- View: User portfolio summary (for /portfolio command)
CREATE OR REPLACE VIEW user_portfolio_summary AS
SELECT
    user_id_hash,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') AS completed_trades,
    COUNT(*) FILTER (WHERE status IN ('OPEN', 'PARTIAL')) AS active_trades,
    SUM(pnl) FILTER (WHERE status = 'COMPLETED') AS total_pnl,
    ROUND(AVG(pnl) FILTER (WHERE status = 'COMPLETED')::numeric, 2) AS avg_pnl,
    COUNT(*) FILTER (WHERE status = 'COMPLETED' AND pnl > 0) AS winning_trades,
    COUNT(*) FILTER (WHERE status = 'COMPLETED' AND pnl <= 0) AS losing_trades,
    MIN(created_at) AS first_trade_at,
    MAX(created_at) AS last_trade_at
FROM completed_trades
GROUP BY user_id_hash;

COMMENT ON VIEW user_portfolio_summary IS 'Aggregate portfolio statistics per user for /portfolio command';

-- View: Recent user trades (for /history command)
CREATE OR REPLACE VIEW user_recent_trades AS
SELECT
    ct.id,
    ct.user_id_hash,
    ct.item_id,
    ct.item_name,
    ct.buy_price,
    ct.sell_price,
    ct.quantity,
    ct.status,
    ct.pnl,
    ct.predicted_fill_probability,
    ct.actual_filled,
    ct.created_at,
    ct.completed_at,
    EXTRACT(EPOCH FROM (ct.completed_at - ct.created_at)) / 3600 AS duration_hours
FROM completed_trades ct
WHERE ct.created_at > NOW() - INTERVAL '30 days'
ORDER BY ct.created_at DESC;

COMMENT ON VIEW user_recent_trades IS 'Recent trades for /history command (last 30 days)';

-- View: ML feedback export (for model calibration)
CREATE OR REPLACE VIEW ml_feedback_export AS
SELECT
    ct.id,
    ct.item_id,
    ct.item_name,
    ct.model_id,
    ct.prediction_id,
    ct.hour_offset,
    ct.offset_pct,
    ct.predicted_fill_probability,
    ct.actual_filled,
    ct.pnl,
    ct.created_at,
    ct.completed_at,
    mr.run_id AS model_run_id,
    mr.mean_auc AS model_auc,
    mr.status AS model_status
FROM completed_trades ct
LEFT JOIN model_registry mr ON mr.id = ct.model_id
WHERE ct.status = 'COMPLETED'
  AND ct.actual_filled IS NOT NULL;

COMMENT ON VIEW ml_feedback_export IS 'Completed trades with prediction data for ML feedback and calibration';

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_completed_trades_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS completed_trades_updated_at ON completed_trades;
CREATE TRIGGER completed_trades_updated_at
    BEFORE UPDATE ON completed_trades
    FOR EACH ROW
    EXECUTE FUNCTION update_completed_trades_updated_at();

COMMENT ON FUNCTION update_completed_trades_updated_at IS 'Auto-updates updated_at timestamp on row modification';

COMMIT;
