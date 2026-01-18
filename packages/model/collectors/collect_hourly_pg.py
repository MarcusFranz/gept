#!/usr/bin/env python3
"""Hourly Price Collector - PostgreSQL version."""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psycopg2
from psycopg2.extras import execute_values
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pybreaker import CircuitBreaker, CircuitBreakerError
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
DATA_DIR = os.getenv("DATA_DIR", "/data")

METRICS_PORT = int(os.getenv("METRICS_PORT", "9101"))
COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", "3600"))
USER_AGENT = "GePT-HourlyCollector/3.0 (PostgreSQL)"

API_TIMESERIES = "https://prices.runescape.wiki/api/v1/osrs/timeseries"

HIGH_VOLUME_ITEMS = [
    2, 314, 453, 554, 555, 556, 557, 558, 560, 561, 562, 563, 564, 565, 566,
    1391, 1513, 1515, 1747, 1749, 1751, 2359, 2361, 2363, 2434, 2485, 3024,
    207, 209, 211, 213, 215, 217, 219, 225, 449, 451, 536, 1617, 1619, 1621,
    1623, 1631, 1637, 5295, 5300, 5304, 5316, 6685, 8778, 8782, 9144, 11232,
    11237, 12934, 13190, 21817, 21820, 21930, 22124, 24598, 24607
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REQUESTS_TOTAL = Counter("gept_hourly_requests_total", "Total API requests", ["status"])
ITEMS_COLLECTED = Counter("gept_hourly_items_total", "Total hourly records collected")
COLLECTION_DURATION = Histogram("gept_hourly_duration_seconds", "Collection duration")
LAST_COLLECTION = Gauge("gept_hourly_last_timestamp", "Last successful collection")
DB_RECONNECT_SUCCESS = Counter("gept_hourly_db_reconnect_success_total", "Successful DB reconnections")
DB_RECONNECT_FAILURES = Counter("gept_hourly_db_reconnect_failures_total", "Failed DB reconnection attempts")
API_RETRIES = Counter("gept_hourly_api_retries_total", "API request retry attempts")
CIRCUIT_OPEN = Counter("gept_hourly_circuit_open_total", "Times circuit breaker opened")
CIRCUIT_STATE = Gauge("gept_hourly_circuit_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)")

# Data quality metrics
dq_metrics = get_data_quality_metrics('hourly')

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
    name="wiki_hourly_api",
    listeners=[
        (_on_circuit_open, 'open'),
        (_on_circuit_close, 'close'),
        (_on_circuit_half_open, 'half_open'),
    ]
)

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


def get_items_to_fetch(conn) -> list:
    items_file = Path(DATA_DIR) / "high_volume_items.json"
    if items_file.exists():
        with open(items_file) as f:
            data = json.load(f); items = data.get("item_ids", data) if isinstance(data, dict) else data
    else:
        items = HIGH_VOLUME_ITEMS
    
    with conn.cursor() as cur:
        # Use tuple for IN clause
        cur.execute(f"""
            SELECT item_id, MAX(timestamp) as latest
            FROM prices_1h
            WHERE item_id IN ({','.join(str(i) for i in items)})
            GROUP BY item_id
        """)
        results = cur.fetchall()
        latest_map = {row[0]: row[1] for row in results}
    
    now = datetime.now(timezone.utc)
    stale_items = []
    for item_id in items:
        latest = latest_map.get(item_id)
        if latest is None or (now - latest).total_seconds() > 7200:
            stale_items.append(item_id)
    
    return stale_items


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
def _fetch_timeseries_with_retry(item_id: int) -> dict:
    """Internal function with retry logic."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            API_TIMESERIES,
            params={"timestep": "1h", "id": item_id},
            headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
        return resp.json()

def fetch_timeseries(item_id: int) -> list:
    """Fetch timeseries data with circuit breaker protection."""
    try:
        data = wiki_api_breaker.call(_fetch_timeseries_with_retry, item_id)
        REQUESTS_TOTAL.labels(status="success").inc()
        return data.get("data", [])
    except CircuitBreakerError:
        logger.warning("Circuit breaker is open, skipping API call")
        REQUESTS_TOTAL.labels(status="circuit_open").inc()
        return []
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        logger.error(f"Failed to fetch item {item_id} after retries: {e}")
        return []


def store_hourly(conn, item_id: int, data: list) -> int:
    if not data:
        return 0

    rows = []
    null_high_count = 0
    null_low_count = 0

    for record in data:
        ts = datetime.fromtimestamp(record["timestamp"], tz=timezone.utc)
        avg_high = record.get("avgHighPrice")
        avg_low = record.get("avgLowPrice")

        # Track null values
        if avg_high is None:
            null_high_count += 1
        if avg_low is None:
            null_low_count += 1

        rows.append((
            item_id, ts,
            avg_high,
            record.get("highPriceVolume", 0),
            avg_low,
            record.get("lowPriceVolume", 0)
        ))

    # Record null value metrics
    if null_high_count > 0:
        dq_metrics.record_null_values('avg_high_price', null_high_count)
    if null_low_count > 0:
        dq_metrics.record_null_values('avg_low_price', null_low_count)

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO prices_1h
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                VALUES %s
                ON CONFLICT (item_id, timestamp) DO NOTHING
                """,
                rows
            )
            inserted = cur.rowcount
            # Track duplicates
            duplicates = len(rows) - inserted
            if duplicates > 0:
                dq_metrics.record_duplicates(duplicates)
            conn.commit()
    except psycopg2.Error:
        dq_metrics.record_db_commit_failure()
        raise

    ITEMS_COLLECTED.inc(inserted)
    return inserted


def run_collection(conn) -> int:
    items = get_items_to_fetch(conn)
    logger.info(f"Fetching {len(items)} items needing updates")

    total_inserted = 0
    for i, item_id in enumerate(items):
        if shutdown_requested:
            break

        # Track per-item collection duration
        with dq_metrics.time_item():
            data = fetch_timeseries(item_id)
            inserted = store_hourly(conn, item_id, data)
        total_inserted += inserted

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(items)} items, {total_inserted} new rows")

        time.sleep(0.1)

    return total_inserted


def main():
    logger.info(f"Starting hourly PostgreSQL collector (interval={COLLECTION_INTERVAL}s)")
    
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server on port {METRICS_PORT}")
    
    conn = get_db_connection()
    logger.info("Connected to PostgreSQL")
    
    while not shutdown_requested:
        try:
            start = time.time()
            inserted = run_collection(conn)
            duration = time.time() - start
            COLLECTION_DURATION.observe(duration)
            LAST_COLLECTION.set(time.time())
            logger.info(f"Collection complete: {inserted} new rows in {duration:.1f}s")
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
        
        sleep_time = max(0, COLLECTION_INTERVAL - (time.time() - start))
        logger.info(f"Sleeping {sleep_time:.0f}s until next collection")
        
        while sleep_time > 0 and not shutdown_requested:
            time.sleep(min(sleep_time, 10))
            sleep_time -= 10
    
    conn.close()
    logger.info("Collector stopped")


if __name__ == "__main__":
    main()
