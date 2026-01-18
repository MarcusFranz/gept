#!/usr/bin/env python3
"""1-Minute Latest Price Collector - High-frequency tick data.

Fetches latest prices every 60 seconds from OSRS Wiki API.
Stores to PostgreSQL/TimescaleDB prices_latest table.
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import psycopg2
from psycopg2.extras import execute_values
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from shared.metrics import get_data_quality_metrics
from shared.health import HealthChecker, add_health_routes_to_prometheus

# Reconnection settings
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 2  # seconds

# Circuit breaker settings
CIRCUIT_FAIL_MAX = 5  # Number of failures before opening
CIRCUIT_RESET_TIMEOUT = 60  # Seconds before attempting recovery

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "host.docker.internal")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "osrs_data")
DB_USER = os.getenv("DB_USER", "osrs_user")
DB_PASS = os.getenv("DB_PASS", "osrs_pass")

METRICS_PORT = int(os.getenv("METRICS_PORT", "9103"))
COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", "60"))  # 1 minute
USER_AGENT = "GePT-Collector/2.0 (https://github.com/gept)"

API_LATEST = "https://prices.runescape.wiki/api/v1/osrs/latest"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "gept_latest_requests_total", "Total API requests", ["status"]
)
ITEMS_COLLECTED = Counter(
    "gept_latest_items_total", "Total items collected"
)
COLLECTION_DURATION = Histogram(
    "gept_latest_duration_seconds", "Collection duration"
)
LAST_COLLECTION = Gauge(
    "gept_latest_last_timestamp", "Last successful collection"
)
DB_RECONNECT_SUCCESS = Counter(
    "gept_latest_db_reconnect_success_total", "Successful DB reconnections"
)
DB_RECONNECT_FAILURES = Counter(
    "gept_latest_db_reconnect_failures_total", "Failed DB reconnection attempts"
)
API_RETRIES = Counter(
    "gept_latest_api_retries_total", "API request retry attempts"
)
CIRCUIT_OPEN = Counter(
    "gept_latest_circuit_open_total", "Times circuit breaker opened"
)
CIRCUIT_STATE = Gauge(
    "gept_latest_circuit_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)"
)

# Data quality metrics
dq_metrics = get_data_quality_metrics('latest')

# Health checker for /health endpoint
health_checker = HealthChecker(
    service_name='latest-1m',
    collection_interval=COLLECTION_INTERVAL,
    last_collection_gauge=LAST_COLLECTION,
)
health_checker.set_api_url(API_LATEST)

# Circuit breaker for Wiki API
def _on_circuit_open(breaker, ex):
    """Called when circuit breaker opens."""
    CIRCUIT_OPEN.inc()
    CIRCUIT_STATE.set(1)
    logger.warning(f"Circuit breaker opened after {CIRCUIT_FAIL_MAX} failures: {ex}")

def _on_circuit_close(breaker):
    """Called when circuit breaker closes."""
    CIRCUIT_STATE.set(0)
    logger.info("Circuit breaker closed, API calls resumed")

def _on_circuit_half_open(breaker):
    """Called when circuit breaker enters half-open state."""
    CIRCUIT_STATE.set(2)
    logger.info("Circuit breaker half-open, testing API...")

wiki_api_breaker = CircuitBreaker(
    fail_max=CIRCUIT_FAIL_MAX,
    reset_timeout=CIRCUIT_RESET_TIMEOUT,
    name="wiki_latest_api",
    listeners=[
        (_on_circuit_open, 'open'),
        (_on_circuit_close, 'close'),
        (_on_circuit_half_open, 'half_open'),
    ]
)

# -----------------------------------------------------------------------------
# Graceful Shutdown
# -----------------------------------------------------------------------------
shutdown_requested = False

def signal_handler(signum: int, frame: Any) -> None:
    global shutdown_requested
    logger.info("Shutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prices_latest_1m (
                timestamp TIMESTAMPTZ NOT NULL,
                item_id INTEGER NOT NULL,
                high INTEGER,
                high_time TIMESTAMPTZ,
                low INTEGER,
                low_time TIMESTAMPTZ,
                PRIMARY KEY (item_id, timestamp)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_latest_1m_ts 
            ON prices_latest_1m (timestamp DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_latest_1m_item_ts 
            ON prices_latest_1m (item_id, timestamp DESC)
        """)
        conn.commit()
    logger.info("Table prices_latest_1m ready")

# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------
def _log_retry(retry_state):
    """Log retry attempts and increment metrics."""
    API_RETRIES.inc()
    logger.warning(f"API request failed, retrying (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
    before_sleep=_log_retry,
    reraise=True
)
def _fetch_latest_impl() -> dict:
    """Internal fetch implementation with retry logic."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(API_LATEST, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()

def fetch_latest() -> dict:
    """Fetch latest prices with circuit breaker protection."""
    try:
        data = wiki_api_breaker.call(_fetch_latest_impl)
        REQUESTS_TOTAL.labels(status="success").inc()
        return data
    except CircuitBreakerError:
        logger.warning("Circuit breaker is open, skipping API call")
        REQUESTS_TOTAL.labels(status="circuit_open").inc()
        raise

def store_latest(conn, data: dict):
    now = datetime.now(timezone.utc)
    rows = []
    null_high_count = 0
    null_low_count = 0

    for item_id_str, prices in data.get("data", {}).items():
        item_id = int(item_id_str)
        high = prices.get("high")
        high_time = prices.get("highTime")
        low = prices.get("low")
        low_time = prices.get("lowTime")

        # Track null values
        if high is None:
            null_high_count += 1
        if low is None:
            null_low_count += 1

        high_ts = datetime.fromtimestamp(high_time, tz=timezone.utc) if high_time else None
        low_ts = datetime.fromtimestamp(low_time, tz=timezone.utc) if low_time else None

        rows.append((now, item_id, high, high_ts, low, low_ts))

    # Record null value metrics
    if null_high_count > 0:
        dq_metrics.record_null_values('high', null_high_count)
    if null_low_count > 0:
        dq_metrics.record_null_values('low', null_low_count)

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO prices_latest_1m (timestamp, item_id, high, high_time, low, low_time)
                VALUES %s
                ON CONFLICT (item_id, timestamp) DO NOTHING
                """,
                rows,
            )
            # Track duplicates
            duplicates = len(rows) - cur.rowcount
            if duplicates > 0:
                dq_metrics.record_duplicates(duplicates)
            conn.commit()
    except psycopg2.Error:
        dq_metrics.record_db_commit_failure()
        raise

    ITEMS_COLLECTED.inc(len(rows))
    return len(rows)

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    logger.info(f"Starting 1-minute latest collector (interval={COLLECTION_INTERVAL}s)")

    # Add /health endpoint to Prometheus server
    add_health_routes_to_prometheus(health_checker)

    # Start Prometheus metrics server (now also serves /health)
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server on port {METRICS_PORT} (also serves /health)")
    
    # Connect to database
    conn = get_db_connection()
    ensure_table(conn)

    # Register DB connection with health checker
    health_checker.set_db_connection(conn)

    while not shutdown_requested:
        try:
            start = time.time()
            
            logger.info("Fetching latest prices...")
            data = fetch_latest()
            
            count = store_latest(conn, data)
            
            duration = time.time() - start
            COLLECTION_DURATION.observe(duration)
            LAST_COLLECTION.set(time.time())
            
            logger.info(f"Stored {count} latest prices in {duration:.2f}s")
            
        except CircuitBreakerError:
            # Circuit breaker is open, just skip this cycle
            pass
        except psycopg2.OperationalError as e:
            logger.error(f"Database error: {e}")
            conn = None
            health_checker.set_db_connection(None)  # Mark DB as disconnected
            for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                delay = min(30, RECONNECT_BASE_DELAY ** attempt)
                logger.info(f"Reconnection attempt {attempt}/{MAX_RECONNECT_ATTEMPTS} in {delay}s...")
                time.sleep(delay)
                try:
                    conn = get_db_connection()
                    health_checker.set_db_connection(conn)  # Update health checker
                    logger.info("Database reconnection successful")
                    DB_RECONNECT_SUCCESS.inc()
                    break
                except Exception as reconnect_error:
                    logger.error(f"Reconnection attempt {attempt} failed: {reconnect_error}")
                    DB_RECONNECT_FAILURES.inc()
            else:
                logger.critical("Max reconnection attempts reached, exiting")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Collection error: {e}")
            REQUESTS_TOTAL.labels(status="error").inc()

        # Sleep until next interval
        sleep_time = max(0, COLLECTION_INTERVAL - (time.time() - start))
        time.sleep(sleep_time)
    
    conn.close()
    logger.info("Collector stopped")

if __name__ == "__main__":
    main()
