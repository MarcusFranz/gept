-- Migration: Add offset_pct attribution to trades tables
-- Run against Neon database

ALTER TABLE active_trades
ADD COLUMN IF NOT EXISTS offset_pct DECIMAL(5, 4);

ALTER TABLE trade_history
ADD COLUMN IF NOT EXISTS offset_pct DECIMAL(5, 4);

COMMENT ON COLUMN active_trades.offset_pct IS 'Offset percentage used for the recommendation that created the trade (0.0125-0.0250)';
COMMENT ON COLUMN trade_history.offset_pct IS 'Offset percentage used for the recommendation that created the trade (0.0125-0.0250)';

