-- Migration: 001_create_trade_outcomes_table
-- Description: Creates the trade_outcomes table for ML feedback loop
-- Issue: #17 - Add endpoint to report trade outcomes for model improvement
-- Date: 2026-01-10

-- Create trade_outcomes table for ML feedback loop
CREATE TABLE IF NOT EXISTS trade_outcomes (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,      -- SHA256 hash of Discord user ID
    rec_id VARCHAR(50) NOT NULL,             -- Recommendation ID from the engine
    item_id INTEGER NOT NULL,                -- OSRS item ID
    item_name VARCHAR(100) NOT NULL,         -- Item name at time of trade
    buy_price INTEGER NOT NULL,              -- Actual buy price (gp)
    sell_price INTEGER NOT NULL,             -- Actual sell price (gp)
    quantity INTEGER NOT NULL,               -- Quantity traded
    actual_profit BIGINT NOT NULL,           -- Actual profit/loss (gp)
    reported_at TIMESTAMP WITH TIME ZONE NOT NULL,  -- When user reported outcome
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying during model training
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_item_id ON trade_outcomes(item_id);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_user_id_hash ON trade_outcomes(user_id_hash);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_rec_id ON trade_outcomes(rec_id);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_reported_at ON trade_outcomes(reported_at);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_created_at ON trade_outcomes(created_at);

-- Comment for documentation
COMMENT ON TABLE trade_outcomes IS 'Stores user-reported trade outcomes for ML model feedback loop';
COMMENT ON COLUMN trade_outcomes.user_id_hash IS 'SHA256 hash of Discord user ID - never store raw Discord IDs';
COMMENT ON COLUMN trade_outcomes.rec_id IS 'Recommendation ID from the prediction engine';
COMMENT ON COLUMN trade_outcomes.actual_profit IS 'Actual profit/loss in gp (negative for losses)';
COMMENT ON COLUMN trade_outcomes.reported_at IS 'Timestamp when user reported the outcome via Discord bot';
