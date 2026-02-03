-- Add use_beta_model toggle to users table
-- Allows users to opt into predictions from a shadow/experimental model
ALTER TABLE users ADD COLUMN IF NOT EXISTS use_beta_model BOOLEAN NOT NULL DEFAULT false;
