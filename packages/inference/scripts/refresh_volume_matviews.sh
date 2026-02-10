#!/bin/bash
# Refresh volume materialized views (called after inference cycle)
# Uses CONCURRENTLY to avoid blocking reads during refresh.
set -euo pipefail

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-osrs_data}"
DB_USER="${DB_USER:-osrs_user}"

PSQL="psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -v ON_ERROR_STOP=1"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Refreshing volume materialized views..."

if $PSQL -tAc "SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_volume_24h'" | grep -q 1; then
    $PSQL -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_volume_24h;"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   mv_volume_24h refreshed"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   WARN: mv_volume_24h does not exist, skipping"
fi

if $PSQL -tAc "SELECT 1 FROM pg_matviews WHERE matviewname = 'mv_volume_1h'" | grep -q 1; then
    $PSQL -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_volume_1h;"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   mv_volume_1h refreshed"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   WARN: mv_volume_1h does not exist, skipping"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Volume materialized view refresh complete"
