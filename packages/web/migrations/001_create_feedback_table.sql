-- Create feedback table for user feedback system
-- Run this migration: psql $DATABASE_URL -f migrations/001_create_feedback_table.sql

CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN ('bug', 'feature', 'general', 'recommendation')),
  rating TEXT CHECK (rating IN ('positive', 'negative')),
  message TEXT,
  email TEXT,
  -- Recommendation context (only for type='recommendation')
  rec_id TEXT,
  rec_item_id INTEGER,
  rec_item_name TEXT,
  rec_buy_price BIGINT,
  rec_sell_price BIGINT,
  rec_quantity INTEGER,
  rec_expected_profit BIGINT,
  rec_confidence REAL,
  rec_model_id TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Index for querying by user
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);

-- Index for querying by type
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(type);

-- Index for querying recommendations with ratings
CREATE INDEX IF NOT EXISTS idx_feedback_rec ON feedback(rec_id) WHERE type = 'recommendation';
