#!/bin/bash
# =============================================================================
# GePT Database Backup Script
# =============================================================================
# Backs up critical tables to compressed SQL files.
# Designed to run daily via cron.
#
# Usage: ./backup_database.sh
# Cron:  0 2 * * * /home/ubuntu/gept/scripts/backup_database.sh >> /home/ubuntu/gept/logs/backup.log 2>&1
# =============================================================================

set -euo pipefail

# Error handler for debugging
error_handler() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Script failed on line $1" >&2
    exit 1
}
trap 'error_handler $LINENO' ERR

BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="osrs_data"
DB_USER="osrs_user"
DB_HOST="localhost"
DB_PORT="5432"

# Export password for pg_dump (required)
if [ -z "$DB_PASS" ]; then
    echo "ERROR: DB_PASS environment variable is required"
    exit 1
fi
export PGPASSWORD="$DB_PASS"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Alert stub (integrate with Discord/PagerDuty later)
send_alert() {
    local level="$1"
    local message="$2"
    log "$level: $message"
    # TODO: Add Discord webhook, PagerDuty integration
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Define backup file path
BACKUP_FILE="$BACKUP_DIR/critical_tables_$DATE.sql.gz"

log "Starting backup..."

# Backup predictions and calibration tables (smaller, critical)
log "Backing up critical tables..."
pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" \
    --table=predictions \
    --table=actual_fills \
    --table=calibration_metrics \
    --table=items \
    --compress=9 \
    --file="$BACKUP_FILE"

# Set restrictive permissions (owner read/write only)
chmod 600 "$BACKUP_FILE"

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log "Backup created: $(basename "$BACKUP_FILE") ($BACKUP_SIZE)"

# Verify gzip integrity
log "Verifying backup integrity..."
if ! gzip -t "$BACKUP_FILE"; then
    log "ERROR: Backup file is corrupted!"
    rm -f "$BACKUP_FILE"
    send_alert "CRITICAL" "Backup verification failed - file corrupted"
    exit 1
fi
log "Integrity check passed"

# Check minimum backup size (previous backups are ~2-5MB)
MIN_SIZE_KB=1000  # 1MB minimum
ACTUAL_SIZE_KB=$(du -k "$BACKUP_FILE" | cut -f1)
if [ "$ACTUAL_SIZE_KB" -lt "$MIN_SIZE_KB" ]; then
    log "WARNING: Backup suspiciously small: ${ACTUAL_SIZE_KB}KB (min: ${MIN_SIZE_KB}KB)"
    send_alert "WARNING" "Backup size unusually small: ${ACTUAL_SIZE_KB}KB"
fi

# Weekly restore test (Mondays only)
if [ "$(date +%u)" -eq 1 ]; then
    log "Running weekly restore test..."
    TEST_DB="osrs_backup_test_$(date +%Y%m%d)"

    # Create test database
    if createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$TEST_DB" 2>/dev/null; then
        # Restore backup
        if gunzip -c "$BACKUP_FILE" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$TEST_DB" > /dev/null 2>&1; then
            # Verify row counts
            EXPECTED_ITEMS=4500
            ACTUAL_ITEMS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -t -A -c "SELECT COUNT(*) FROM items" "$TEST_DB" 2>/dev/null || echo "0")

            if [ "$ACTUAL_ITEMS" -lt "$EXPECTED_ITEMS" ]; then
                log "ERROR: Restore test failed - items count: $ACTUAL_ITEMS (expected ~$EXPECTED_ITEMS)"
                send_alert "CRITICAL" "Backup restore test failed - items count mismatch"
            else
                log "Restore test passed - items count: $ACTUAL_ITEMS"
            fi
        else
            log "ERROR: Failed to restore backup to test database"
            send_alert "CRITICAL" "Backup restore test failed - restore error"
        fi

        # Cleanup test database
        dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$TEST_DB" 2>/dev/null || true
    else
        log "WARNING: Could not create test database - skipping restore test"
    fi
fi

# Keep only last 7 days of backups
log "Cleaning old backups..."
find "$BACKUP_DIR" -name "critical_tables_*.sql.gz" -mtime +7 -delete

# List remaining backups
log "Current backups:"
ls -lh "$BACKUP_DIR"/critical_tables_*.sql.gz 2>/dev/null || echo "No backups found"

log "Backup verified and complete!"
