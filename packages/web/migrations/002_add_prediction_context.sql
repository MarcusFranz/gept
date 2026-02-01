-- Add prediction context columns to active_trades and trade_history
-- These fields capture model prediction data for future ML feedback loop
-- Run: psql $DATABASE_URL -f migrations/002_add_prediction_context.sql

-- Active trades: store prediction context when trade is created from recommendation
ALTER TABLE active_trades ADD COLUMN IF NOT EXISTS confidence TEXT;
ALTER TABLE active_trades ADD COLUMN IF NOT EXISTS fill_probability REAL;
ALTER TABLE active_trades ADD COLUMN IF NOT EXISTS expected_profit BIGINT;

-- Trade history: preserve prediction context when trade completes
ALTER TABLE trade_history ADD COLUMN IF NOT EXISTS expected_profit BIGINT;
ALTER TABLE trade_history ADD COLUMN IF NOT EXISTS confidence TEXT;
ALTER TABLE trade_history ADD COLUMN IF NOT EXISTS fill_probability REAL;
ALTER TABLE trade_history ADD COLUMN IF NOT EXISTS expected_hours REAL;
