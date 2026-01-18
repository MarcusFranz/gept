# GePT Data Collection Services

Production data collection infrastructure for OSRS Grand Exchange market data.

## Overview

This directory contains all data collection services that continuously gather market data from the OSRS Wiki API and store it in PostgreSQL/TimescaleDB. These services run on the Ampere A1 server and provide the foundational data for the GePT ML prediction system.

## Architecture

```
OSRS Wiki API → Docker Collectors → PostgreSQL/TimescaleDB → ML Pipeline
```

### Active Services (Docker)

| Service | Script | Interval | Port | Status |
|---------|--------|----------|------|--------|
| 1m Latest Prices | `collect_latest_1m.py` | 60s | 9103 | ✅ Active |
| 5m OHLC Prices | `collect_5m_pg.py` | 300s | 9100 | ✅ Active |
| Hourly Prices | `collect_hourly_pg.py` | 3600s | 9101 | ✅ Active |
| News Feed | `collect_news_pg.py` | 1800s | 9102 | ✅ Active |
| Dashboard | `dashboard.py` | - | 8080 | ✅ Active |

### Legacy Services (Systemd)

| Service | Script | Status |
|---------|--------|--------|
| Player Counts | `collect_player_counts.py` | ✅ Active (DuckDB) |
| Item Metadata | `update_items.py` | ✅ Active (DuckDB) |
| Legacy Collector | `osrs_collector_pg.py` | ⚠️ Redundant |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- PostgreSQL 14+ with TimescaleDB
- Python 3.10+

### Deployment

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

2. **Deploy with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Verify Services**:
   ```bash
   docker-compose ps
   curl http://localhost:8080  # Dashboard
   curl http://localhost:9100/metrics  # Prometheus metrics
   ```

### Monitoring

- **Dashboard**: http://localhost:8080
- **Prometheus Metrics**: Ports 9100-9103
- **Logs**: `docker-compose logs -f [service-name]`

## Service Details

### 1. 1-Minute Latest Price Collector

**Purpose**: High-frequency tick data for real-time price tracking

**API Endpoint**: `https://prices.runescape.wiki/api/v1/osrs/latest`

**Table**: `prices_latest_1m`
```sql
CREATE TABLE prices_latest_1m (
    timestamp TIMESTAMPTZ NOT NULL,
    item_id INTEGER NOT NULL,
    high INTEGER,
    high_time TIMESTAMPTZ,
    low INTEGER,
    low_time TIMESTAMPTZ,
    PRIMARY KEY (item_id, timestamp)
);
```

**Features**:
- Collects last trade prices for all tradeable items
- Tracks exact trade timestamps
- Deduplicates on conflict
- Prometheus metrics for monitoring

**Resource Usage**: ~126 MiB RAM, <0.01% CPU

---

### 2. 5-Minute OHLC Collector

**Purpose**: Primary data source for ML model training (426M rows)

**API Endpoint**: `https://prices.runescape.wiki/api/v1/osrs/5m`

**Table**: `price_data_5min` (TimescaleDB hypertable)
```sql
CREATE TABLE price_data_5min (
    item_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    avg_high_price INTEGER,
    high_price_volume INTEGER,
    avg_low_price INTEGER,
    low_price_volume INTEGER
);
```

**Features**:
- 5-minute OHLC averages with volumes
- TimescaleDB compression eligible
- Critical for ML feature engineering
- Handles API timestamp correctly

**Resource Usage**: ~100 MiB RAM, <0.01% CPU

---

### 3. Hourly Timeseries Collector

**Purpose**: Historical data for backtesting and long-term analysis

**API Endpoint**: `https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=1h&id={item_id}`

**Table**: `prices_1h`
```sql
CREATE TABLE prices_1h (
    item_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    avg_high_price INTEGER,
    high_price_volume INTEGER,
    avg_low_price INTEGER,
    low_price_volume INTEGER,
    PRIMARY KEY (item_id, timestamp)
);
```

**Features**:
- Smart fetching: only updates stale items (>2 hours old)
- Focuses on high-volume items (~65 items)
- Rate-limited API calls (0.1s delay)
- Bulk inserts with conflict handling

**Tracked Items**: See `HIGH_VOLUME_ITEMS` in script (runes, logs, ores, popular items)

**Resource Usage**: ~95 MiB RAM, <0.01% CPU

---

### 4. News Collector

**Purpose**: Market event correlation and anomaly detection

**RSS Feeds**:
- Official OSRS news
- Wiki recent changes

**Table**: `osrs_news`
```sql
CREATE TABLE osrs_news (
    guid VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    link TEXT,
    description TEXT,
    category VARCHAR(100),
    pub_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Features**:
- Parses RSS feeds
- Deduplicates by GUID
- Categorizes wiki edits
- Useful for correlating game updates with price movements

**Resource Usage**: ~39 MiB RAM, <0.01% CPU

---

### 5. Dashboard

**Purpose**: Real-time operational monitoring

**Port**: 8080

**Features**:
- Table statistics (row counts, sizes, freshness)
- Container health checks
- Auto-refresh every 5 seconds
- Lightweight HTML/JS interface

**Endpoints**:
- `/` - Web dashboard
- `/api/status` - JSON status
- `/health` - Health check

**Resource Usage**: ~16 MiB RAM, <0.01% CPU

---

### 6. Player Count Collector (Legacy)

**Purpose**: Track concurrent OSRS players

**Source**: Scrapes OSRS homepage HTML

**Storage**: DuckDB (`data/player_counts.duckdb`)

**Status**: ⚠️ Active but uses DuckDB instead of PostgreSQL

**Migration Needed**: Should move to PostgreSQL for unified data access

---

### 7. Item Metadata Updater (Legacy)

**Purpose**: Keep item reference data up-to-date

**API Endpoint**: `https://prices.runescape.wiki/api/v1/osrs/mapping`

**Storage**: DuckDB (`data/items.duckdb`)

**Update Frequency**: Daily (86400s)

**Status**: ⚠️ Active but uses DuckDB instead of PostgreSQL

**Migration Needed**: PostgreSQL already has an `items` table that should be used

---

### 8. Legacy Combined Collector (Deprecated)

**Script**: `osrs_collector_pg.py`

**Status**: ⚠️ **REDUNDANT** - duplicates Docker services

**Recommendation**: Stop this service - functionality covered by Docker containers

---

## Database Schema

### TimescaleDB Hypertables

The `price_data_5min` table should be converted to a TimescaleDB hypertable for optimal performance:

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert to hypertable (if not already)
SELECT create_hypertable('price_data_5min', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

-- Enable compression
ALTER TABLE price_data_5min SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'item_id'
);

-- Add compression policy (compress data older than 7 days)
SELECT add_compression_policy('price_data_5min', INTERVAL '7 days');

-- Add retention policy (optional - keep 90 days of 5m data)
SELECT add_retention_policy('price_data_5min', INTERVAL '90 days');
```

### Indexes

```sql
-- 5-minute prices
CREATE INDEX idx_5m_item_ts ON price_data_5min (item_id, timestamp DESC);
CREATE INDEX idx_5m_ts ON price_data_5min (timestamp DESC);

-- 1-minute latest
CREATE INDEX idx_latest_1m_ts ON prices_latest_1m (timestamp DESC);
CREATE INDEX idx_latest_1m_item_ts ON prices_latest_1m (item_id, timestamp DESC);

-- Hourly prices
CREATE INDEX idx_1h_item_ts ON prices_1h (item_id, timestamp DESC);
CREATE INDEX idx_1h_ts ON prices_1h (timestamp DESC);

-- News
CREATE INDEX idx_news_date ON osrs_news (pub_date DESC);
CREATE INDEX idx_news_category ON osrs_news (category);
```

## Configuration

### Environment Variables

```bash
# Database (PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=osrs_data
DB_USER=osrs_user
DB_PASS=your_password

# Service Configuration
COLLECTION_INTERVAL=300  # seconds
METRICS_PORT=9100
DATA_DIR=/data

# Dashboard
DASHBOARD_PORT=8080
```

### Docker Compose Override

For development, create `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  ge-5m-collector:
    environment:
      COLLECTION_INTERVAL: "60"  # Faster collection for testing
```

## Monitoring & Metrics

### Prometheus Metrics

Each collector exposes Prometheus metrics:

**Common Metrics**:
- `gept_*_requests_total{status="success|error"}` - API request counter
- `gept_*_items_total` - Items collected counter
- `gept_*_duration_seconds` - Collection duration histogram
- `gept_*_last_timestamp` - Last successful collection timestamp

**Example Prometheus Scrape Config**:
```yaml
scrape_configs:
  - job_name: 'gept-collectors'
    static_configs:
      - targets:
        - 'localhost:9100'  # 5m collector
        - 'localhost:9101'  # hourly collector
        - 'localhost:9102'  # news collector
        - 'localhost:9103'  # 1m collector
```

### Health Checks

All Docker services have HTTP health checks on their metrics ports:

```bash
# Check 5m collector
curl http://localhost:9100/metrics

# Check if healthy via Docker
docker inspect osrs-5m-collector --format='{{.State.Health.Status}}'
```

### Alerting Rules (Recommended)

```yaml
groups:
  - name: gept_collectors
    rules:
      - alert: CollectorStale
        expr: time() - gept_5m_last_timestamp > 900
        annotations:
          summary: "5m collector hasn't collected in 15 minutes"

      - alert: CollectorErrors
        expr: rate(gept_5m_requests_total{status="error"}[5m]) > 0.1
        annotations:
          summary: "5m collector error rate > 10%"
```

## Maintenance

### Backup Strategy

**Database Backup** (daily):
```bash
pg_dump -h localhost -U osrs_user osrs_data \
  --compress=9 \
  --file="backup-$(date +%Y%m%d).sql.gz"
```

**Table-Specific Backup** (for large tables):
```bash
pg_dump -h localhost -U osrs_user osrs_data \
  --table=price_data_5min \
  --compress=9 \
  --file="prices_5m-$(date +%Y%m%d).sql.gz"
```

### Disk Space Management

**Current Usage**: 79GB / 98GB (85%)

**Immediate Actions**:
1. Enable TimescaleDB compression (can save 50-90%)
2. Implement retention policies
3. Archive old data to cloud storage

**Compression Example**:
```sql
-- Compress all chunks older than 7 days
SELECT compress_chunk(i, if_not_compressed => true)
FROM show_chunks('price_data_5min', older_than => INTERVAL '7 days') i;

-- Check compression status
SELECT * FROM timescaledb_information.compressed_chunk_stats;
```

### Troubleshooting

**Problem**: Collector not inserting data

**Checks**:
1. Check logs: `docker-compose logs [service-name]`
2. Test API manually: `curl https://prices.runescape.wiki/api/v1/osrs/5m`
3. Check DB connection: `docker exec [container] pg_isready -h DB_HOST`
4. Check metrics: `curl http://localhost:9100/metrics | grep error`

**Problem**: Database connection errors

**Solution**:
- Verify PostgreSQL is running: `systemctl status postgresql`
- Check connection: `psql -h localhost -U osrs_user osrs_data`
- Docker networking: Use `host.docker.internal` for localhost from containers

**Problem**: Disk full

**Solution**:
- Check table sizes: `SELECT * FROM pg_size_pretty(pg_total_relation_size('price_data_5min'));`
- Enable compression (see above)
- Clean up Docker logs: `docker system prune`

## Development

### Local Testing

1. **Set up local PostgreSQL**:
   ```bash
   docker run -d \
     -e POSTGRES_DB=osrs_data \
     -e POSTGRES_USER=osrs_user \
     -e POSTGRES_PASSWORD=dev_password \
     -p 5432:5432 \
     timescale/timescaledb:latest-pg14
   ```

2. **Run collector locally**:
   ```bash
   export DB_HOST=localhost
   export DB_PASS=dev_password
   python collect_5m_pg.py
   ```

3. **Test with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

### Adding a New Collector

1. Create new Python script in `collectors/`
2. Follow existing patterns:
   - Logging setup
   - Prometheus metrics
   - Graceful shutdown (SIGTERM/SIGINT)
   - Database connection with retry
   - Environment variable configuration
3. Add to `docker-compose.yml`
4. Update this README
5. Test locally before deploying

### Code Structure

**Recommended pattern**:
```python
import logging
import os
import signal
from prometheus_client import start_http_server, Counter
import psycopg2

# Configuration from environment
DB_HOST = os.getenv("DB_HOST", "localhost")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUESTS = Counter("gept_requests_total", "Requests", ["status"])

# Graceful shutdown
shutdown_requested = False
def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Main loop
def main():
    start_http_server(METRICS_PORT)
    conn = psycopg2.connect(...)

    while not shutdown_requested:
        try:
            # Collection logic
            pass
        except Exception as e:
            logger.error(f"Error: {e}")
            REQUESTS.labels(status="error").inc()

        time.sleep(COLLECTION_INTERVAL)

    conn.close()
```

## Performance Optimization

### Current Performance (Production)

| Metric | Value |
|--------|-------|
| Total CPU Usage | <0.5% |
| Total Memory | ~500 MB |
| API Latency | ~100-300ms |
| DB Insert Rate | ~1000 rows/min |
| Disk I/O | <1 MB/s |

**Assessment**: Highly efficient, no optimization needed

### Potential Optimizations (if needed)

1. **Batch Inserts**: Already implemented with `execute_values()`
2. **Connection Pooling**: Could use `psycopg2.pool` for high-frequency collectors
3. **Async I/O**: Could use `asyncio` + `asyncpg` for parallel fetching
4. **Caching**: Could cache item metadata to reduce DB queries

**Note**: Current performance is excellent; optimizations not required.

## Security

### Current Security Posture

**Strengths**:
- Non-root Docker containers
- Isolated Docker network
- Rate limiting on API calls
- Health checks for monitoring

**Improvements Needed**:
1. Move DB credentials to secrets management
2. Firewall metrics ports (9100-9103)
3. Add TLS to dashboard (or put behind nginx)
4. Implement API key rotation for external APIs (when required)

### Secrets Management (Recommended)

Use Docker secrets or environment files:

```bash
# .env (gitignored)
DB_PASS=your_secure_password
API_KEY=your_api_key
```

Or use Docker secrets:
```bash
echo "your_password" | docker secret create db_password -
```

## Deployment Checklist

Before deploying to production:

- [ ] Environment variables configured
- [ ] Database credentials secured
- [ ] TimescaleDB compression enabled
- [ ] Retention policies set
- [ ] Prometheus scraping configured
- [ ] Alerting rules created
- [ ] Backup strategy implemented
- [ ] Firewall rules configured
- [ ] Dashboard accessible
- [ ] Health checks passing
- [ ] Logs rotating properly
- [ ] Disk space monitored

## Support & Maintenance

**Maintainer**: Marcus Franz (@marcusfranz)

**Issues**: Report to GitHub issues

**Logs Location**:
- Docker: `docker-compose logs`
- Systemd: `/home/ubuntu/osrs_collector/*.log`

**Monitoring**: `http://$AMPERE_IP:8080` (dashboard, see `config/servers.env` for IP)

## License

Proprietary - GePT Project

## References

- [OSRS Wiki API Documentation](https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
