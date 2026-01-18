-- Migration 011: Cleanup Zombie Training Jobs
-- Mark stuck training jobs as FAILED

-- Cleanup jobs stuck in intermediate states for more than 1 day
-- These are zombie jobs that will never complete
UPDATE training_jobs
SET status = 'FAILED',
    error_message = 'Marked as failed - stuck in intermediate state for >24h',
    status_changed_at = NOW()
WHERE status IN ('SELECTING', 'TRAINING', 'PREPARING', 'VALIDATING', 'DEPLOYING')
  AND status_changed_at < NOW() - INTERVAL '1 day';

-- Add index for faster status lookups (if not exists)
CREATE INDEX IF NOT EXISTS idx_training_jobs_status_changed
ON training_jobs (status, status_changed_at);

COMMENT ON INDEX idx_training_jobs_status_changed IS 'Speeds up zombie job detection queries';
