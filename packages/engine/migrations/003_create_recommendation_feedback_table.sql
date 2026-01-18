-- Migration: 003_create_recommendation_feedback_table
-- Description: Creates the recommendation_feedback table for user feedback collection
-- Issue: #137 - Collect and store structured user feedback on recommendations
-- Date: 2026-01-15

-- Create recommendation_feedback table for user feedback on recommendations
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,           -- SHA256 hash of Discord user ID
    rec_id VARCHAR(50),                           -- Recommendation ID (nullable for generic feedback)
    item_id INTEGER NOT NULL,                     -- OSRS item ID
    item_name VARCHAR(100) NOT NULL,              -- Item name at time of feedback
    feedback_type VARCHAR(30) NOT NULL,           -- Structured feedback category
    side VARCHAR(4),                              -- Which side had the issue (buy/sell)
    notes TEXT,                                   -- Optional free-text notes (max 500 chars enforced by API)
    recommended_price INTEGER,                    -- Price from recommendation (optional)
    actual_price INTEGER,                         -- What user actually saw/got (optional)
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,  -- When user submitted feedback
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Constraint to validate feedback_type values
ALTER TABLE recommendation_feedback
ADD CONSTRAINT check_feedback_type
CHECK (feedback_type IN (
    'price_too_high',
    'price_too_low',
    'volume_too_low',
    'filled_quickly',
    'filled_slowly',
    'did_not_fill',
    'spread_too_wide',
    'price_manipulation',
    'other'
));

-- Constraint to validate side values
ALTER TABLE recommendation_feedback
ADD CONSTRAINT check_side
CHECK (side IS NULL OR side IN ('buy', 'sell'));

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_feedback_item_id ON recommendation_feedback(item_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id_hash ON recommendation_feedback(user_id_hash);
CREATE INDEX IF NOT EXISTS idx_feedback_rec_id ON recommendation_feedback(rec_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON recommendation_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_submitted_at ON recommendation_feedback(submitted_at);

-- Comments for documentation
COMMENT ON TABLE recommendation_feedback IS 'Stores structured user feedback on recommendations for ML improvement';
COMMENT ON COLUMN recommendation_feedback.user_id_hash IS 'SHA256 hash of Discord user ID - never store raw Discord IDs';
COMMENT ON COLUMN recommendation_feedback.rec_id IS 'Recommendation ID (rec_{item_id}_{YYYYMMDDHH}) - optional';
COMMENT ON COLUMN recommendation_feedback.feedback_type IS 'Structured feedback category from predefined list';
COMMENT ON COLUMN recommendation_feedback.notes IS 'Optional free-text notes (limited to 500 chars by API)';
