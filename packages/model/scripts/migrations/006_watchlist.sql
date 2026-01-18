-- Migration 006: Watchlist Table
-- Allows users to track items without trading
--
-- Run: psql -h localhost -U osrs_user -d osrs_data -f scripts/migrations/006_watchlist.sql

BEGIN;

CREATE TABLE IF NOT EXISTS watchlist (
    id BIGSERIAL PRIMARY KEY,

    -- User identification (hashed for privacy)
    user_id_hash TEXT NOT NULL,

    -- Item reference
    item_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,

    -- Optional user note
    note TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE (user_id_hash, item_id)
);

-- Comments
COMMENT ON TABLE watchlist IS 'User watchlist for tracking items without trading';
COMMENT ON COLUMN watchlist.user_id_hash IS 'SHA-256 hash of Discord user ID for privacy';
COMMENT ON COLUMN watchlist.item_id IS 'OSRS item ID from items table';
COMMENT ON COLUMN watchlist.item_name IS 'Denormalized item name for display';
COMMENT ON COLUMN watchlist.note IS 'Optional user note about why they are watching this item';

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_watchlist_user
    ON watchlist(user_id_hash, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_watchlist_item
    ON watchlist(item_id);

-- View: User watchlist summary (for /watchlist command)
CREATE OR REPLACE VIEW user_watchlist_summary AS
SELECT
    user_id_hash,
    COUNT(*) AS item_count,
    MIN(created_at) AS oldest_watch,
    MAX(created_at) AS newest_watch
FROM watchlist
GROUP BY user_id_hash;

COMMENT ON VIEW user_watchlist_summary IS 'Aggregate watchlist statistics per user';

COMMIT;
