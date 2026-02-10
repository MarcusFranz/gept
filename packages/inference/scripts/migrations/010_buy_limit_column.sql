-- Migration 010: Add buy_limit column to items table
-- GE buy limits are fetched from Wiki API 'limit' field

-- Add buy_limit column if it doesn't exist
ALTER TABLE items ADD COLUMN IF NOT EXISTS buy_limit INTEGER;

-- Index for queries filtering by buy limit
CREATE INDEX IF NOT EXISTS idx_items_buy_limit
ON items (buy_limit)
WHERE buy_limit IS NOT NULL;

COMMENT ON COLUMN items.buy_limit IS 'Grand Exchange buy limit per 4 hours';
