-- Migration: Add price stability and momentum fields for smarter filtering
-- Run: psql $DATABASE_URL -f packages/model/scripts/migrations/015_add_stability_fields.sql

BEGIN;

-- Add new columns to predictions table
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS median_14d DECIMAL(12,2);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS price_vs_median_ratio DECIMAL(6,4);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS return_1h DECIMAL(6,4);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS return_4h DECIMAL(6,4);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS return_24h DECIMAL(6,4);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS volatility_24h DECIMAL(6,4);

-- Add same columns to staging table if it exists
ALTER TABLE IF EXISTS predictions_staging ADD COLUMN IF NOT EXISTS median_14d DECIMAL(12,2);
ALTER TABLE IF EXISTS predictions_staging ADD COLUMN IF NOT EXISTS price_vs_median_ratio DECIMAL(6,4);
ALTER TABLE IF EXISTS predictions_staging ADD COLUMN IF NOT EXISTS return_1h DECIMAL(6,4);
ALTER TABLE IF EXISTS predictions_staging ADD COLUMN IF NOT EXISTS return_4h DECIMAL(6,4);
ALTER TABLE IF EXISTS predictions_staging ADD COLUMN IF NOT EXISTS return_24h DECIMAL(6,4);
ALTER TABLE IF EXISTS predictions_staging ADD COLUMN IF NOT EXISTS volatility_24h DECIMAL(6,4);

-- Create index for filtering by price anomalies
CREATE INDEX IF NOT EXISTS idx_predictions_stability
ON predictions (time, price_vs_median_ratio, return_4h)
WHERE price_vs_median_ratio IS NOT NULL;

COMMIT;
