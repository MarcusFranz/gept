# Data Collection Services Migration & Optimization Plan

## Overview
Migrate all data collection services from the Ampere server to the local Git repository, optimize infrastructure, and implement best practices for production data collection.

## Critical Files
- Collectors: `collectors/*.py` (24 scripts already copied)
- Database schema: `scripts/setup_predictions_table.sql`
- Deployment: `deploy_ampere.sh`
- Monitoring: `src/monitoring.py`
- Configuration: `collectors/docker-compose.yml`, `collectors/Dockerfile`

## Phase 1: Immediate Actions (Non-Disruptive)

### 1.1 Stop Redundant Legacy Collector
**Goal**: Remove duplicate data collection to reduce resource usage

**Actions**:
- SSH to Ampere server and stop `osrs_collector_pg.py` process (PID 835141)
- Remove from any startup scripts/systemd if configured
- Verify Docker containers continue to handle 5m and latest collections

**Commands**:
```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128 'kill 835141'
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128 'systemctl disable osrs_collector || echo "Not a systemd service"'
```

**Verification**:
- Check Docker containers are still healthy: `docker ps`
- Verify data still flowing: `SELECT MAX(timestamp) FROM price_data_5min`
- Monitor for 1 hour to ensure no gaps in data collection

**Risk**: Low - functionality is duplicated in Docker containers

---

### 1.2 Fix Dashboard Health Check
**Goal**: Resolve "unhealthy" status on dashboard container

**Root Cause**: Health check likely failing due to incorrect endpoint or timing

**Actions**:
- SSH to server and inspect dashboard logs: `docker logs osrs-dashboard`
- Test health endpoint manually: `curl http://localhost:8080/health`
- Update `docker-compose.yml` health check configuration:
  - Change from `/metrics` to `/health` endpoint (dashboard uses `/health`)
  - Increase interval/timeout if needed

**File to Edit**: `collectors/docker-compose.yml`
```yaml
dashboard:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]  # Changed from /metrics
    interval: 30s
    timeout: 10s
    retries: 3
```

**Verification**:
- Run `docker inspect osrs-dashboard --format='{{.State.Health.Status}}'`
- Should return "healthy" within 30 seconds

**Risk**: Low - just a monitoring fix, no functional impact

---

## Phase 2: Database Optimizations (Critical for Disk Space)

### 2.1 Enable TimescaleDB Compression
**Goal**: Reduce disk usage by 50-90% on historical data (currently 79GB/98GB = 85% full)

**Impact**: Can save 20-50GB of disk space immediately

**Actions**:
1. Create compression script: `scripts/enable_timescaledb_compression.sql`
2. Enable compression on `price_data_5min` table
3. Set up compression policy (compress data older than 7 days)
4. Add retention policy (optional - keep 90 days of 5m data, keep hourly forever)

**Script Content**:
```sql
-- Enable TimescaleDB extension (if not already)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert price_data_5min to hypertable if not already
SELECT create_hypertable('price_data_5min', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Enable compression
ALTER TABLE price_data_5min SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'item_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy (compress chunks older than 7 days)
SELECT add_compression_policy('price_data_5min', INTERVAL '7 days', if_not_exists => TRUE);

-- Optional: Add retention policy (delete data older than 90 days)
-- SELECT add_retention_policy('price_data_5min', INTERVAL '90 days', if_not_exists => TRUE);

-- Manually compress existing old chunks (one-time operation)
SELECT compress_chunk(i, if_not_compressed => true)
FROM show_chunks('price_data_5min', older_than => INTERVAL '7 days') i;

-- Check compression status
SELECT * FROM timescaledb_information.compressed_chunk_stats
WHERE hypertable_name = 'price_data_5min';
```

**Execution**:
```bash
# Via SSH tunnel
ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128
# In another terminal
psql -h localhost -U osrs_user -d osrs_data -f scripts/enable_timescaledb_compression.sql
```

**Verification**:
- Query compression stats: `SELECT * FROM timescaledb_information.compressed_chunk_stats;`
- Check disk usage: `SELECT pg_size_pretty(pg_total_relation_size('price_data_5min'));`
- Verify queries still work: `SELECT COUNT(*) FROM price_data_5min WHERE timestamp > NOW() - INTERVAL '24 hours';`

**Risk**: Medium - requires testing queries on compressed data, but TimescaleDB handles this transparently

**Rollback**: `SELECT decompress_chunk(i) FROM show_chunks('price_data_5min') i;`

---

### 2.2 Create Backup Strategy
**Goal**: Protect data with automated backups before making changes

**Actions**:
1. Create backup script: `scripts/backup_database.sh`
2. Test backup and restore process
3. Set up automated daily backups via cron

**Script Content** (`scripts/backup_database.sh`):
```bash
#!/bin/bash
set -e

BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="osrs_data"
DB_USER="osrs_user"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup predictions and calibration tables (smaller, critical)
pg_dump -h localhost -U $DB_USER $DB_NAME \
    --table=predictions \
    --table=actual_fills \
    --table=calibration_metrics \
    --table=items \
    --compress=9 \
    --file="$BACKUP_DIR/critical_tables_$DATE.sql.gz"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "critical_tables_*.sql.gz" -mtime +7 -delete

echo "Backup completed: critical_tables_$DATE.sql.gz"
```

**Cron Setup**:
```bash
# Daily at 2 AM
0 2 * * * /home/ubuntu/gept/scripts/backup_database.sh >> /home/ubuntu/gept/logs/backup.log 2>&1
```

**Verification**:
- Test restore: `gunzip -c backup.sql.gz | psql -h localhost -U osrs_user osrs_data_test`
- Check backup file size is reasonable

**Risk**: Low - read-only operation

---

## Phase 3: Service Consolidation (Unified Storage)

### 3.1 Migrate Player Count Collector to PostgreSQL
**Goal**: Consolidate DuckDB data into PostgreSQL for unified access

**Current State**:
- Script: `collectors/collect_player_counts.py`
- Storage: DuckDB file (`data/player_counts.duckdb`)
- Status: Active systemd service

**Actions**:
1. Create PostgreSQL table for player counts
2. Migrate existing DuckDB data to PostgreSQL
3. Update collector script to use PostgreSQL
4. Deploy updated collector
5. Verify data collection continues
6. Archive DuckDB file

**New Table Schema**:
```sql
CREATE TABLE IF NOT EXISTS player_counts (
    timestamp TIMESTAMPTZ NOT NULL PRIMARY KEY,
    count INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_player_counts_ts ON player_counts (timestamp DESC);
```

**Migration Script** (`scripts/migrate_player_counts.py`):
```python
import duckdb
import psycopg2
from psycopg2.extras import execute_values

# Connect to DuckDB
duck_conn = duckdb.connect('/home/ubuntu/osrs_collector/data/player_counts.duckdb')

# Connect to PostgreSQL
pg_conn = psycopg2.connect(
    host='localhost',
    port=5432,
    dbname='osrs_data',
    user='osrs_user',
    password='$DB_PASS'
)

# Extract data from DuckDB
rows = duck_conn.execute("SELECT timestamp, count FROM player_counts ORDER BY timestamp").fetchall()

# Insert into PostgreSQL
with pg_conn.cursor() as cur:
    execute_values(
        cur,
        "INSERT INTO player_counts (timestamp, count) VALUES %s ON CONFLICT (timestamp) DO NOTHING",
        rows
    )
    pg_conn.commit()

print(f"Migrated {len(rows)} player count records")

duck_conn.close()
pg_conn.close()
```

**Updated Collector** (`collectors/collect_player_counts_pg.py`):
- Replace DuckDB connection with PostgreSQL connection
- Use environment variables for DB credentials (match pattern from other collectors)
- Keep same collection logic (60-second interval, HTML scraping)
- Add Prometheus metrics for consistency

**Deployment**:
1. Test locally with environment variables
2. Deploy to server via SCP
3. Update systemd service to use new script
4. Restart service
5. Monitor for 1 hour

**Verification**:
- Query PostgreSQL: `SELECT COUNT(*) FROM player_counts;`
- Check latest timestamp: `SELECT MAX(timestamp) FROM player_counts;`
- Verify new data arriving every 60 seconds

**Risk**: Low - can run in parallel with existing DuckDB collector initially

---

### 3.2 Migrate Item Metadata Updater to PostgreSQL
**Goal**: Use existing PostgreSQL `items` table instead of DuckDB

**Current State**:
- Script: `collectors/update_items.py`
- Storage: DuckDB file (`data/items.duckdb`)
- PostgreSQL table: `items` already exists with 4,500 items

**Actions**:
1. Verify `items` table schema matches API response structure
2. Update collector script to use PostgreSQL
3. Deploy and test
4. Archive DuckDB file

**Updated Script** (`collectors/update_items_pg.py`):
```python
# Change connection from DuckDB to PostgreSQL
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "osrs_data"),
        user=os.getenv("DB_USER", "osrs_user"),
        password=os.getenv("DB_PASS", "$DB_PASS")
    )

# Update table creation/upsert logic for PostgreSQL
def update_items(conn, items_data):
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO items (id, name, members, limit_ge, value, highalch, lowalch, examine, icon, last_updated)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                members = EXCLUDED.members,
                limit_ge = EXCLUDED.limit_ge,
                value = EXCLUDED.value,
                highalch = EXCLUDED.highalch,
                lowalch = EXCLUDED.lowalch,
                examine = EXCLUDED.examine,
                icon = EXCLUDED.icon,
                last_updated = NOW()
            """,
            [(item['id'], item['name'], item.get('members'), ...) for item in items_data]
        )
        conn.commit()
```

**Verification**:
- Query item count: `SELECT COUNT(*) FROM items;`
- Check last update time: `SELECT MAX(last_updated) FROM items;`
- Verify new items appear after API updates

**Risk**: Low - table already exists and is used by other parts of the system

---

## Phase 4: Code Organization & Refactoring

### 4.1 Create Shared Database Utility Module
**Goal**: DRY principle - eliminate hardcoded credentials across codebase

**Current Problem**:
- Password `$DB_PASS` appears in 9+ files
- Connection logic duplicated across collectors and ML pipeline

**Solution**: Create `src/db_utils.py` with unified connection management

**Implementation**:

**File**: `src/db_utils.py`
```python
import os
import psycopg2
from psycopg2 import pool
from typing import Optional

# Default credentials (can be overridden by environment)
DEFAULT_DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'dbname': os.getenv('DB_NAME', 'osrs_data'),
    'user': os.getenv('DB_USER', 'osrs_user'),
    'password': os.getenv('DB_PASS', '$DB_PASS')  # Will move to env vars
}

# Connection pool (singleton)
_connection_pool: Optional[pool.ThreadedConnectionPool] = None

def get_connection():
    """Get a database connection from the pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **DEFAULT_DB_CONFIG
        )
    return _connection_pool.getconn()

def release_connection(conn):
    """Release a connection back to the pool."""
    if _connection_pool:
        _connection_pool.putconn(conn)

def get_simple_connection():
    """Get a simple connection (not from pool) for long-running operations."""
    return psycopg2.connect(**DEFAULT_DB_CONFIG)
```

**Refactor All Files to Use `db_utils`**:

Files to update:
1. `src/batch_predictor_fast.py` - replace ConnectionPool class
2. `src/monitoring.py` - replace CONN_PARAMS
3. `src/evaluation_job.py` - replace direct connection
4. `training/train_production_catboost.py` - replace DB_CONFIG
5. `training/export_training_data.py` - replace DB_CONFIG
6. All collector scripts - replace individual connection logic

**Example Refactor** (`src/batch_predictor_fast.py`):
```python
# Old:
from psycopg2 import pool
CONN_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'osrs_data',
    'user': 'osrs_user',
    'password': '$DB_PASS'
}

# New:
from src.db_utils import get_connection, release_connection

# Usage
conn = get_connection()
try:
    # ... do work ...
finally:
    release_connection(conn)
```

**Benefits**:
- Single source of truth for DB credentials
- Easy to add environment variable support
- Connection pooling available to all components
- Easier to test with mock connections

**Risk**: Medium - requires testing all affected components

**Testing Strategy**:
1. Update one file at a time
2. Test inference pipeline: `python run_inference.py --dry-run`
3. Test monitoring: `python src/monitoring.py --check all`
4. Test training export: `python training/export_training_data.py --help`
5. Test collectors locally before deploying

---

### 4.2 Create Shared Prometheus Metrics Module
**Goal**: Standardize metrics collection across all collectors

**File**: `collectors/shared/metrics.py`
```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

class CollectorMetrics:
    """Standard metrics for data collectors."""

    def __init__(self, service_name: str):
        self.requests_total = Counter(
            f'gept_{service_name}_requests_total',
            'Total API requests',
            ['status']
        )
        self.items_collected = Counter(
            f'gept_{service_name}_items_total',
            'Total items collected'
        )
        self.collection_duration = Histogram(
            f'gept_{service_name}_duration_seconds',
            'Collection duration'
        )
        self.last_collection = Gauge(
            f'gept_{service_name}_last_timestamp',
            'Last successful collection'
        )

    def start_server(self, port: int):
        """Start Prometheus metrics HTTP server."""
        start_http_server(port)
```

**Usage in Collectors**:
```python
from shared.metrics import CollectorMetrics

metrics = CollectorMetrics('5m')
metrics.start_server(9100)

# In collection loop
with metrics.collection_duration.time():
    data = fetch_data()
    metrics.requests_total.labels(status='success').inc()
    metrics.items_collected.inc(len(data))
    metrics.last_collection.set(time.time())
```

**Benefits**:
- Consistent metric naming across all collectors
- Reduced code duplication
- Easier to add new metrics

---

### 4.3 Create Shared Configuration Module
**Goal**: Centralized configuration management

**File**: `collectors/shared/config.py`
```python
import os
from typing import Optional

class CollectorConfig:
    """Configuration for data collectors."""

    # Database
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_NAME: str = os.getenv('DB_NAME', 'osrs_data')
    DB_USER: str = os.getenv('DB_USER', 'osrs_user')
    DB_PASS: str = os.getenv('DB_PASS', '$DB_PASS')

    # API
    USER_AGENT: str = os.getenv('USER_AGENT', 'GePT-Collector/3.0 (https://github.com/gept)')

    # Intervals
    INTERVAL_5M: int = int(os.getenv('COLLECTION_INTERVAL_5M', '300'))
    INTERVAL_1M: int = int(os.getenv('COLLECTION_INTERVAL_1M', '60'))
    INTERVAL_HOURLY: int = int(os.getenv('COLLECTION_INTERVAL_HOURLY', '3600'))
    INTERVAL_NEWS: int = int(os.getenv('COLLECTION_INTERVAL_NEWS', '1800'))

    @classmethod
    def get_db_params(cls) -> dict:
        """Get database connection parameters."""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'dbname': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASS
        }
```

---

## Phase 5: Monitoring & Alerting Enhancements

### 5.1 Set Up Prometheus Server
**Goal**: Centralize metrics collection from all collectors

**Actions**:
1. Install Prometheus on Ampere server (Docker container)
2. Configure scrape targets for all collector metrics endpoints
3. Set up persistent storage for metrics
4. Configure retention policy (30 days)

**Docker Compose Addition** (`collectors/docker-compose.yml`):
```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: gept-prometheus
    restart: unless-stopped
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
```

**Prometheus Config** (`collectors/prometheus/prometheus.yml`):
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gept-collectors'
    static_configs:
      - targets:
        - 'osrs-5m-collector:9100'
        - 'osrs-hourly-collector:9101'
        - 'osrs-news-collector:9102'
        - 'osrs-latest-1m:9103'
    metrics_path: '/metrics'
```

**Verification**:
- Access Prometheus UI: http://150.136.170.128:9090
- Query: `gept_5m_items_total`
- Check targets: http://150.136.170.128:9090/targets

---

### 5.2 Set Up Grafana Dashboards
**Goal**: Visual monitoring of all collectors

**Actions**:
1. Deploy Grafana container
2. Connect to Prometheus data source
3. Create dashboard for collector metrics
4. Create dashboard for database metrics

**Docker Compose Addition**:
```yaml
  grafana:
    image: grafana/grafana:latest
    container_name: gept-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin  # Change in production
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
```

**Dashboard Panels**:
- Collection rate (items/minute)
- API error rate
- Collection duration (P50, P95, P99)
- Database connection pool utilization
- Disk usage trends
- Prediction freshness

---

### 5.3 Implement Alerting Rules
**Goal**: Proactive notification of issues

**Prometheus Alert Rules** (`collectors/prometheus/alerts.yml`):
```yaml
groups:
  - name: gept_collectors
    interval: 1m
    rules:
      - alert: CollectorStale
        expr: time() - gept_5m_last_timestamp > 900
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "5m collector hasn't collected data in 15 minutes"

      - alert: CollectorHighErrorRate
        expr: rate(gept_5m_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "5m collector error rate > 10%"

      - alert: DiskSpaceHigh
        expr: node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes < 0.15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data disk <15% free space"
```

**Alertmanager Config** (for Discord/Slack/Email):
```yaml
receivers:
  - name: 'discord'
    webhook_configs:
      - url: 'DISCORD_WEBHOOK_URL'
        send_resolved: true
```

---

## Phase 6: Deployment Automation

### 6.1 Create Unified Deployment Script
**Goal**: Single command to deploy all services

**File**: `deploy_collectors.sh`
```bash
#!/bin/bash
set -e

SERVER="ubuntu@150.136.170.128"
SSH_KEY=".secrets/oracle_key.pem"
REMOTE_DIR="/home/ubuntu/osrs_collector"

echo "Deploying GePT Data Collectors..."

# 1. Sync collector scripts
echo "Syncing collector scripts..."
rsync -avz -e "ssh -i $SSH_KEY" \
    collectors/*.py \
    collectors/docker-compose.yml \
    collectors/Dockerfile \
    collectors/requirements.txt \
    $SERVER:$REMOTE_DIR/

# 2. Sync shared modules
echo "Syncing shared modules..."
rsync -avz -e "ssh -i $SSH_KEY" \
    collectors/shared/ \
    $SERVER:$REMOTE_DIR/shared/

# 3. Sync Prometheus/Grafana configs
echo "Syncing monitoring configs..."
rsync -avz -e "ssh -i $SSH_KEY" \
    collectors/prometheus/ \
    $SERVER:$REMOTE_DIR/prometheus/

# 4. Restart Docker containers
echo "Restarting Docker containers..."
ssh -i $SSH_KEY $SERVER "cd $REMOTE_DIR && docker-compose up -d --build"

# 5. Verify health
echo "Verifying health..."
sleep 10
ssh -i $SSH_KEY $SERVER "docker-compose ps"
ssh -i $SSH_KEY $SERVER "curl -s http://localhost:8080/health"

echo "Deployment complete!"
```

**Usage**: `./deploy_collectors.sh`

---

### 6.2 Create Service Status Script
**Goal**: Quick health check of all services

**File**: `scripts/check_collectors.sh`
```bash
#!/bin/bash

SERVER="ubuntu@150.136.170.128"
SSH_KEY=".secrets/oracle_key.pem"

echo "=== GePT Collector Status ==="
echo

# Docker containers
echo "Docker Containers:"
ssh -i $SSH_KEY $SERVER 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
echo

# Database freshness
echo "Data Freshness:"
ssh -i $SSH_KEY $SERVER "psql -h localhost -U osrs_user -d osrs_data -c \"
SELECT
    '5m prices' as table_name,
    MAX(timestamp) as latest,
    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) as seconds_old
FROM price_data_5min
UNION ALL
SELECT
    'predictions' as table_name,
    MAX(time) as latest,
    EXTRACT(EPOCH FROM (NOW() - MAX(time))) as seconds_old
FROM predictions;
\""
echo

# Disk usage
echo "Disk Usage:"
ssh -i $SSH_KEY $SERVER 'df -h /data'
```

**Usage**: `./scripts/check_collectors.sh`

---

## Phase 7: Documentation & Testing

### 7.1 Update CLAUDE.md
**Goal**: Update project instructions with new collector information

**Additions**:
```markdown
## Data Collection Services

All data collection services are in the `collectors/` directory:

- **5-Minute Collector**: Collects OHLC prices every 5 minutes
- **1-Minute Collector**: High-frequency tick data
- **Hourly Collector**: Historical hourly data
- **News Collector**: Market news and wiki changes
- **Player Count Collector**: Active player tracking
- **Item Metadata**: Item properties and GE limits

### Deployment

Deploy all collectors:
```bash
./deploy_collectors.sh
```

Check status:
```bash
./scripts/check_collectors.sh
```

### Monitoring

- Dashboard: http://150.136.170.128:8080
- Prometheus: http://150.136.170.128:9090
- Grafana: http://150.136.170.128:3000
```

---

### 7.2 Create Testing Checklist

**Pre-Deployment Testing**:
- [ ] Test all collectors locally with environment variables
- [ ] Verify database connections work
- [ ] Test Prometheus metrics endpoints
- [ ] Run docker-compose up locally
- [ ] Test health checks
- [ ] Verify logs are formatted correctly

**Post-Deployment Verification**:
- [ ] All Docker containers show "healthy" status
- [ ] Metrics endpoints responding (ports 9100-9103)
- [ ] Dashboard shows green status
- [ ] Database tables receiving new data
- [ ] No gaps in data collection
- [ ] Prometheus scraping all targets
- [ ] Grafana dashboards showing data
- [ ] Alert rules loaded correctly

**7-Day Monitoring Period**:
- [ ] No data collection gaps
- [ ] No unexpected errors in logs
- [ ] Disk compression working as expected
- [ ] Backup cron running daily
- [ ] All metrics within expected ranges

---

## Implementation Timeline

### Week 1: Foundation & Safety
- Day 1: Stop redundant collector, fix dashboard health check
- Day 2: Enable TimescaleDB compression (with testing)
- Day 3: Set up backup strategy
- Day 4: Create `db_utils.py` and refactor 2-3 files
- Day 5: Testing and verification

### Week 2: Consolidation
- Day 6-7: Migrate player count collector to PostgreSQL
- Day 8-9: Migrate item metadata to PostgreSQL
- Day 10: Create shared modules (metrics, config)

### Week 3: Monitoring
- Day 11-12: Deploy Prometheus and Grafana
- Day 13: Create dashboards
- Day 14: Set up alerting rules
- Day 15: Testing and documentation

### Week 4: Deployment & Polish
- Day 16-17: Create deployment automation
- Day 18: Comprehensive testing
- Day 19: Update documentation
- Day 20: Final verification and handoff

---

## Risk Mitigation

**Database Changes (Compression)**:
- Backup before enabling compression
- Test on dev database first if available
- Monitor query performance after compression
- Have rollback plan (decompress chunks)

**Service Migration (DuckDB → PostgreSQL)**:
- Run new and old collectors in parallel initially
- Verify data matches before switching over
- Keep DuckDB files for 30 days as backup

**Code Refactoring (db_utils)**:
- Update one component at a time
- Test each component before moving to next
- Keep git history clean with clear commit messages

**Production Deployment**:
- Deploy during low-traffic periods (if applicable)
- Have SSH access ready for quick rollback
- Monitor logs actively for 1 hour post-deployment

---

## Success Criteria

1. **Disk Space**: Usage drops from 85% to <60% after compression
2. **Service Consolidation**: All services using PostgreSQL (no DuckDB)
3. **Code Quality**: No hardcoded credentials, DRY principles followed
4. **Monitoring**: Prometheus + Grafana dashboards operational
5. **Reliability**: Zero data collection gaps during migration
6. **Documentation**: Complete and up-to-date README and CLAUDE.md
7. **Automation**: Single-command deployment working

---

## Rollback Plans

**If compression causes issues**:
```sql
SELECT decompress_chunk(i) FROM show_chunks('price_data_5min') i;
```

**If new collectors fail**:
- Revert to previous docker-compose.yml
- Restart old DuckDB collectors via systemd

**If database refactoring breaks inference**:
- Git revert to previous commit
- Redeploy with `deploy_ampere.sh`

---

## Maintenance After Implementation

**Daily**:
- Check dashboard for green status
- Verify backup completed successfully

**Weekly**:
- Review Grafana dashboards for trends
- Check disk space usage
- Review any alerts that fired

**Monthly**:
- Review and update high-volume items list
- Check compression effectiveness
- Update dependencies if needed
- Review and optimize slow queries

---

## Notes

- All changes are designed to be non-disruptive
- Data collection continues throughout migration
- Existing inference engine is not affected
- Can be implemented incrementally over 4 weeks
- Each phase can be tested independently
