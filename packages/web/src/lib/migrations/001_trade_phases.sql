-- Migration: Add phase tracking to active_trades
-- Run against Neon database

ALTER TABLE active_trades
ADD COLUMN IF NOT EXISTS phase VARCHAR(10) DEFAULT 'buying' CHECK (phase IN ('buying', 'selling')),
ADD COLUMN IF NOT EXISTS progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
ADD COLUMN IF NOT EXISTS last_check_in TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS next_check_in TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS actual_buy_price INTEGER,
ADD COLUMN IF NOT EXISTS actual_sell_price INTEGER,
ADD COLUMN IF NOT EXISTS expected_hours DECIMAL(4,1);

-- Index for check-in queries
CREATE INDEX IF NOT EXISTS idx_active_trades_next_check_in
ON active_trades(user_id, next_check_in)
WHERE next_check_in IS NOT NULL;

COMMENT ON COLUMN active_trades.phase IS 'Current trade phase: buying or selling';
COMMENT ON COLUMN active_trades.progress IS 'User-reported fill progress 0-100';
COMMENT ON COLUMN active_trades.last_check_in IS 'When user last reported progress';
COMMENT ON COLUMN active_trades.next_check_in IS 'When next check-in is due';
COMMENT ON COLUMN active_trades.actual_buy_price IS 'Actual price if relisted (null = used recommendation)';
COMMENT ON COLUMN active_trades.actual_sell_price IS 'Actual sell price if adjusted';
COMMENT ON COLUMN active_trades.expected_hours IS 'Expected fill time from recommendation';
