#!/bin/bash
# Run all database migrations for the training pipeline
#
# Usage:
#   ./scripts/migrations/run_migrations.sh              # Run via SSH tunnel
#   ./scripts/migrations/run_migrations.sh --local      # Run on Ampere server directly
#
# Prerequisites:
#   - SSH tunnel: ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 $AMPERE_HOST
#     (AMPERE_HOST is defined in config/servers.env, default: ubuntu@150.136.170.128)
#   - Or run directly on Ampere server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="$SCRIPT_DIR"

# Database connection
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-osrs_data}"
DB_USER="${DB_USER:-osrs_user}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for password
if [ -z "${DB_PASS:-}" ]; then
    if [ -f "$SCRIPT_DIR/../../.secrets/db_pass" ]; then
        DB_PASS=$(cat "$SCRIPT_DIR/../../.secrets/db_pass")
    else
        log_error "DB_PASS not set and .secrets/db_pass not found"
        echo "Set DB_PASS environment variable or create .secrets/db_pass file"
        exit 1
    fi
fi

export PGPASSWORD="$DB_PASS"

# Test connection
log_info "Testing database connection..."
if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
    log_error "Cannot connect to database"
    echo "Ensure SSH tunnel is running: ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 \$AMPERE_HOST"
    echo "(AMPERE_HOST is defined in config/servers.env)"
    exit 1
fi
log_info "Database connection successful"

# Run migrations in order
MIGRATIONS=(
    "001_model_registry.sql"
    "002_training_jobs.sql"
    "003_model_performance.sql"
    "004_predictions_model_fk.sql"
    "005_completed_trades.sql"
    "006_watchlist.sql"
)

log_info "Running ${#MIGRATIONS[@]} migrations..."

for migration in "${MIGRATIONS[@]}"; do
    migration_path="$MIGRATIONS_DIR/$migration"

    if [ ! -f "$migration_path" ]; then
        log_error "Migration file not found: $migration"
        exit 1
    fi

    log_info "Running $migration..."
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration_path" -v ON_ERROR_STOP=1; then
        log_info "  ✓ $migration completed"
    else
        log_error "  ✗ $migration failed"
        exit 1
    fi
done

log_info "All migrations completed successfully!"

# Show summary
echo ""
log_info "Database summary:"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT
    'model_registry' as table_name,
    (SELECT COUNT(*) FROM model_registry) as row_count
UNION ALL
SELECT
    'training_jobs',
    (SELECT COUNT(*) FROM training_jobs)
UNION ALL
SELECT
    'model_performance',
    (SELECT COUNT(*) FROM model_performance)
UNION ALL
SELECT
    'training_status',
    (SELECT COUNT(*) FROM training_status)
UNION ALL
SELECT
    'completed_trades',
    (SELECT COUNT(*) FROM completed_trades)
UNION ALL
SELECT
    'watchlist',
    (SELECT COUNT(*) FROM watchlist);
"
