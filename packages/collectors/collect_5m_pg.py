#!/usr/bin/env python3
"""5-Minute Price Collector - PostgreSQL/TimescaleDB version.

Fetches 5-minute OHLC prices from OSRS Wiki API every 5 minutes.
Stores to TimescaleDB price_data_5min hypertable.
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import psycopg2
from psycopg2.extras import execute_values
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pybreaker import CircuitBreaker, CircuitBreakerError, CircuitBreakerListener
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from shared.metrics import get_data_quality_metrics

# Reconnection settings
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 2  # seconds

# Circuit breaker settings
CIRCUIT_FAIL_MAX = 5  # Number of failures before opening
CIRCUIT_RESET_TIMEOUT = 60  # Seconds before attempting recovery

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "osrs_data")
DB_USER = os.getenv("DB_USER", "osrs_user")
DB_PASS = os.environ["DB_PASS"]

METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))
COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", "300"))
USER_AGENT = "GePT-Collector/3.0 (PostgreSQL)"

API_5M = "https://prices.runescape.wiki/api/v1/osrs/5m"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUESTS_TOTAL = Counter("gept_5m_requests_total", "Total API requests", ["status"])
ITEMS_COLLECTED = Counter("gept_5m_items_total", "Total items collected")
COLLECTION_DURATION = Histogram("gept_5m_duration_seconds", "Collection duration")
LAST_COLLECTION = Gauge("gept_5m_last_timestamp", "Last successful collection")
DB_RECONNECT_SUCCESS = Counter("gept_5m_db_reconnect_success_total", "Successful DB reconnections")
DB_RECONNECT_FAILURES = Counter("gept_5m_db_reconnect_failures_total", "Failed DB reconnection attempts")
API_RETRIES = Counter("gept_5m_api_retries_total", "API request retry attempts")
CIRCUIT_OPEN = Counter("gept_5m_circuit_open_total", "Times circuit breaker opened")
CIRCUIT_STATE = Gauge("gept_5m_circuit_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)")

# Data quality metrics
dq_metrics = get_data_quality_metrics('5m')

# Circuit breaker listener for Wiki API
class WikiAPIListener(CircuitBreakerListener):
    """Listener for circuit breaker state changes."""

    def state_change(self, cb, old_state, new_state):
        """Handle state transitions."""
        state_name = new_state.name if hasattr(new_state, 'name') else str(new_state)
        if state_name == 'open':
            CIRCUIT_OPEN.inc()
            CIRCUIT_STATE.set(1)
            logger.warning(f"Circuit breaker opened after {CIRCUIT_FAIL_MAX} failures")
        elif state_name == 'closed':
            CIRCUIT_STATE.set(0)
            logger.info("Circuit breaker closed, API calls resumed")
        elif state_name == 'half-open':
            CIRCUIT_STATE.set(2)
            logger.info("Circuit breaker half-open, testing API...")

wiki_api_breaker = CircuitBreaker(
    fail_max=CIRCUIT_FAIL_MAX,
    reset_timeout=CIRCUIT_RESET_TIMEOUT,
    name="wiki_5m_api",
    listeners=[WikiAPIListener()]
)

# Graceful Shutdown
shutdown_requested = False

def signal_handler(signum: int, frame: Any) -> None:
    global shutdown_requested
    logger.info("Shutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


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
def _fetch_5m_impl() -> dict:
    """Internal fetch implementation with retry logic."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(API_5M, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()

def fetch_5m() -> dict:
    """Fetch 5-minute prices with circuit breaker protection."""
    try:
        data = wiki_api_breaker.call(_fetch_5m_impl)
        REQUESTS_TOTAL.labels(status="success").inc()
        return data
    except CircuitBreakerError:
        logger.warning("Circuit breaker is open, skipping API call")
        REQUESTS_TOTAL.labels(status="circuit_open").inc()
        raise


def validate_price_record(
    item_id: int,
    avg_high: Optional[int],
    high_vol: int,
    avg_low: Optional[int],
    low_vol: int,
    ts: datetime,
) -> Optional[tuple]:
    """Validate and sanitize price data before storage.

    Returns the validated tuple if valid, or None if the record should be skipped.
    """
    # Skip if no price data at all
    if avg_high is None and avg_low is None:
        return None

    # Validate high price range
    if avg_high is not None:
        if avg_high < 0 or avg_high > 2_147_483_647:
            dq_metrics.record_validation_error('invalid_high_price')
            logger.warning(f"Invalid high price for item {item_id}: {avg_high}")
            return None

    # Validate low price range
    if avg_low is not None:
        if avg_low < 0 or avg_low > 2_147_483_647:
            dq_metrics.record_validation_error('invalid_low_price')
            logger.warning(f"Invalid low price for item {item_id}: {avg_low}")
            return None

    # Validate volumes (cannot be negative)
    if high_vol < 0:
        dq_metrics.record_validation_error('negative_high_volume')
        logger.warning(f"Negative high volume for item {item_id}: {high_vol}")
        return None

    if low_vol < 0:
        dq_metrics.record_validation_error('negative_low_volume')
        logger.warning(f"Negative low volume for item {item_id}: {low_vol}")
        return None

    return (item_id, ts, avg_high, high_vol, avg_low, low_vol)


def store_5m(conn, data: dict) -> int:
    api_timestamp = data.get("timestamp")
    if not api_timestamp:
        logger.warning("API response missing timestamp")
        dq_metrics.record_api_validation_failure('missing_timestamp')
        return 0

    ts = datetime.fromtimestamp(api_timestamp, tz=timezone.utc)
    rows = []
    null_high_count = 0
    null_low_count = 0

    for item_id_str, prices in data.get("data", {}).items():
        item_id = int(item_id_str)
        avg_high = prices.get("avgHighPrice")
        high_vol = prices.get("highPriceVolume", 0)
        avg_low = prices.get("avgLowPrice")
        low_vol = prices.get("lowPriceVolume", 0)

        # Track null values for data quality monitoring
        if avg_high is None:
            null_high_count += 1
        if avg_low is None:
            null_low_count += 1

        # Validate and add to rows if valid
        validated = validate_price_record(item_id, avg_high, high_vol, avg_low, low_vol, ts)
        if validated is not None:
            rows.append(validated)

    # Record null value metrics
    if null_high_count > 0:
        dq_metrics.record_null_values('avg_high_price', null_high_count)
    if null_low_count > 0:
        dq_metrics.record_null_values('avg_low_price', null_low_count)

    if not rows:
        return 0

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO price_data_5min
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                VALUES %s
                ON CONFLICT (item_id, timestamp) DO NOTHING
                """,
                rows
            )
            # Track duplicates: rows attempted - rows inserted
            duplicates = len(rows) - cur.rowcount
            if duplicates > 0:
                dq_metrics.record_duplicates(duplicates)
            conn.commit()
    except psycopg2.Error:
        dq_metrics.record_db_commit_failure()
        raise

    ITEMS_COLLECTED.inc(len(rows))
    return len(rows)


def main():
    logger.info(f"Starting 5-minute PostgreSQL collector (interval={COLLECTION_INTERVAL}s)")
    
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server on port {METRICS_PORT}")
    
    conn = get_db_connection()
    logger.info("Connected to PostgreSQL")
    
    while not shutdown_requested:
        try:
            start = time.time()
            
            logger.info("Fetching 5-minute prices...")
            data = fetch_5m()
            
            count = store_5m(conn, data)
            
            duration = time.time() - start
            COLLECTION_DURATION.observe(duration)
            LAST_COLLECTION.set(time.time())
            
            logger.info(f"Stored {count} 5m prices in {duration:.2f}s")
            
        except CircuitBreakerError:
            # Circuit breaker is open, just skip this cycle
            pass
        except psycopg2.OperationalError as e:
            logger.error(f"Database error: {e}")
            conn = None
            for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                delay = min(30, RECONNECT_BASE_DELAY ** attempt)
                logger.info(f"Reconnection attempt {attempt}/{MAX_RECONNECT_ATTEMPTS} in {delay}s...")
                time.sleep(delay)
                try:
                    conn = get_db_connection()
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
        
        sleep_time = max(0, COLLECTION_INTERVAL - (time.time() - start))
        time.sleep(sleep_time)
    
    conn.close()
    logger.info("Collector stopped")


if __name__ == "__main__":
    main()
