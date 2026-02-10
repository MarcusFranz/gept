#!/usr/bin/env python3
"""Gap Detector - Data Gap Detection and Backfill Service.

Detects missing data windows in price_data_5min and triggers backfill from
the Wiki API for gaps <24h old.

Can run as:
- One-shot: Detect and backfill gaps, then exit
- Continuous: Run hourly to monitor and backfill gaps
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

import httpx
import psycopg2
from psycopg2.extras import execute_values
from prometheus_client import start_http_server

# Add shared module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.metrics import MonitorMetrics
from shared.health import HealthChecker, add_health_routes_to_prometheus

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "osrs_data")
DB_USER = os.getenv("DB_USER", "osrs_user")
DB_PASS = os.environ["DB_PASS"]

METRICS_PORT = int(os.getenv("METRICS_PORT", "9107"))
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
BACKFILL_BATCH_SIZE = int(os.getenv("BACKFILL_BATCH_SIZE", "100"))
RUN_INTERVAL = int(os.getenv("RUN_INTERVAL", "3600"))  # 1 hour default

# Wiki API configuration
USER_AGENT = "GePT-GapDetector/1.0 (PostgreSQL)"
API_TIMESERIES = "https://prices.runescape.wiki/api/v1/osrs/timeseries"

# Gap detection configuration
MIN_GAP_DURATION = timedelta(minutes=10)  # Ignore gaps smaller than this
MAX_BACKFILL_AGE = timedelta(hours=24)  # Don't try to backfill gaps older than this

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize metrics
metrics = MonitorMetrics("gap_detector")

# Health checker for /health endpoint
health_checker = HealthChecker(
    service_name='gap-detector',
    collection_interval=RUN_INTERVAL,
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
    """Create a new database connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def get_tracked_items(conn, limit: int = 500) -> List[int]:
    """
    Get list of actively traded items to check for gaps.

    Returns items that have recent price data, prioritized by trading activity.
    """
    query = """
        SELECT DISTINCT item_id
        FROM price_data_5min
        WHERE timestamp > NOW() - INTERVAL '7 days'
        GROUP BY item_id
        HAVING COUNT(*) > 100
        ORDER BY SUM(high_price_volume + low_price_volume) DESC
        LIMIT %s
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            return [row[0] for row in cur.fetchall()]
    except psycopg2.Error as e:
        logger.error(f"Error getting tracked items: {e}")
        return []


def detect_gaps_for_item(
    conn, item_id: int, lookback_hours: int = 24
) -> List[Tuple[datetime, datetime]]:
    """
    Detect gaps in 5-minute data for a specific item.

    Returns list of (gap_start, gap_end) tuples.
    """
    query = """
        WITH timestamps AS (
            SELECT timestamp
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp > NOW() - make_interval(hours => %s)
            ORDER BY timestamp
        ),
        gaps AS (
            SELECT
                timestamp AS gap_start,
                LEAD(timestamp) OVER (ORDER BY timestamp) AS gap_end
            FROM timestamps
        )
        SELECT gap_start, gap_end
        FROM gaps
        WHERE gap_end - gap_start > INTERVAL '10 minutes'
        ORDER BY gap_start DESC
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (item_id, lookback_hours))
            return [(row[0], row[1]) for row in cur.fetchall() if row[1] is not None]
    except psycopg2.Error as e:
        logger.error(f"Error detecting gaps for item {item_id}: {e}")
        return []


def detect_global_gaps(conn, lookback_hours: int = 24) -> List[Tuple[datetime, datetime]]:
    """
    Detect time windows where we have no data for ANY items.

    These indicate systemic collection failures.
    """
    query = """
        WITH expected AS (
            SELECT generate_series(
                date_trunc('hour', NOW() - make_interval(hours => %s)),
                date_trunc('hour', NOW()),
                INTERVAL '5 minutes'
            ) AS expected_ts
        ),
        actual AS (
            SELECT DISTINCT date_trunc('minute', timestamp) AS actual_ts
            FROM price_data_5min
            WHERE timestamp > NOW() - make_interval(hours => %s)
        ),
        missing AS (
            SELECT expected_ts
            FROM expected e
            LEFT JOIN actual a ON date_trunc('minute', e.expected_ts) = a.actual_ts
            WHERE a.actual_ts IS NULL
            ORDER BY expected_ts
        )
        SELECT
            MIN(expected_ts) AS gap_start,
            MAX(expected_ts) + INTERVAL '5 minutes' AS gap_end
        FROM (
            SELECT expected_ts,
                   SUM(CASE WHEN expected_ts - LAG(expected_ts) OVER (ORDER BY expected_ts) > INTERVAL '5 minutes'
                            THEN 1 ELSE 0 END) OVER (ORDER BY expected_ts) AS gap_group
            FROM missing
        ) grouped
        GROUP BY gap_group
        HAVING MAX(expected_ts) - MIN(expected_ts) >= INTERVAL '10 minutes'
        ORDER BY gap_start DESC
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (lookback_hours, lookback_hours))
            return [(row[0], row[1]) for row in cur.fetchall()]
    except psycopg2.Error as e:
        logger.error(f"Error detecting global gaps: {e}")
        return []


def record_gap(
    conn,
    table_name: str,
    item_id: Optional[int],
    gap_start: datetime,
    gap_end: datetime,
) -> int:
    """
    Record a detected gap in the data_quality table.

    Returns the gap ID.
    """
    query = """
        INSERT INTO data_quality (table_name, item_id, gap_start, gap_end, status)
        VALUES (%s, %s, %s, %s, 'DETECTED')
        ON CONFLICT (table_name, item_id, gap_start)
        DO UPDATE SET gap_end = EXCLUDED.gap_end
        RETURNING id
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (table_name, item_id, gap_start, gap_end))
            conn.commit()
            result = cur.fetchone()
            return result[0] if result else 0
    except psycopg2.Error as e:
        logger.error(f"Error recording gap: {e}")
        return 0


def update_gap_status(
    conn,
    gap_id: int,
    status: str,
    backfill_source: Optional[str] = None,
    rows_recovered: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Update the status of a gap record."""
    query = """
        UPDATE data_quality
        SET status = %s,
            backfill_attempts = backfill_attempts + 1,
            backfill_source = COALESCE(%s, backfill_source),
            rows_recovered = rows_recovered + %s,
            error_message = %s,
            resolved_at = CASE WHEN %s IN ('BACKFILLED', 'UNRECOVERABLE', 'IGNORED')
                               THEN NOW() ELSE resolved_at END
        WHERE id = %s
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (status, backfill_source, rows_recovered, error_message, status, gap_id),
            )
            conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error updating gap status: {e}")


def backfill_from_wiki(
    conn, item_id: int, gap_start: datetime, gap_end: datetime
) -> int:
    """
    Backfill missing data from Wiki API timeseries endpoint.

    Returns number of rows inserted.
    """
    try:
        url = f"{API_TIMESERIES}?timestep=5m&id={item_id}"
        headers = {"User-Agent": USER_AGENT}

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        entries = data.get("data", [])
        if not entries:
            logger.warning(f"No timeseries data returned for item {item_id}")
            return 0

        # Filter to gap window
        rows = []
        for entry in entries:
            ts = datetime.fromtimestamp(entry.get("timestamp"), tz=timezone.utc)
            if gap_start <= ts <= gap_end:
                rows.append(
                    (
                        item_id,
                        ts,
                        entry.get("avgHighPrice"),
                        entry.get("highPriceVolume", 0),
                        entry.get("avgLowPrice"),
                        entry.get("lowPriceVolume", 0),
                    )
                )

        if not rows:
            logger.info(f"No data in gap window for item {item_id}")
            return 0

        # Insert with conflict handling
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO price_data_5min
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                VALUES %s
                ON CONFLICT (item_id, timestamp) DO NOTHING
                """,
                rows,
            )
            conn.commit()
            # Get actual rows inserted (not available directly, estimate from returned)
            return len(rows)

    except httpx.HTTPError as e:
        logger.error(f"HTTP error backfilling item {item_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error backfilling item {item_id}: {e}")
        raise


def process_gaps(conn, gaps: List[Tuple[datetime, datetime]], item_id: Optional[int] = None) -> dict:
    """
    Process detected gaps: record them and attempt backfill.

    Returns statistics about processed gaps.
    """
    stats = {
        "detected": 0,
        "backfilled": 0,
        "unrecoverable": 0,
        "errors": 0,
        "rows_recovered": 0,
    }

    now = datetime.now(timezone.utc)
    table_name = "price_data_5min"

    for gap_start, gap_end in gaps:
        stats["detected"] += 1

        # Ensure timezone awareness
        if gap_start.tzinfo is None:
            gap_start = gap_start.replace(tzinfo=timezone.utc)
        if gap_end.tzinfo is None:
            gap_end = gap_end.replace(tzinfo=timezone.utc)

        # Record the gap
        gap_id = record_gap(conn, table_name, item_id, gap_start, gap_end)
        if gap_id == 0:
            stats["errors"] += 1
            continue

        # Record metric
        metrics.record_gap_detected(table_name)

        # Check if gap is too old to backfill
        if now - gap_end > MAX_BACKFILL_AGE:
            logger.warning(
                f"Gap too old to backfill: {gap_start} to {gap_end}"
            )
            update_gap_status(
                conn,
                gap_id,
                "UNRECOVERABLE",
                error_message="Gap older than 24h, Wiki API only provides recent data",
            )
            stats["unrecoverable"] += 1
            continue

        # Attempt backfill
        if item_id is not None:
            try:
                update_gap_status(conn, gap_id, "BACKFILLING", backfill_source="wiki_api")
                rows = backfill_from_wiki(conn, item_id, gap_start, gap_end)

                if rows > 0:
                    update_gap_status(
                        conn, gap_id, "BACKFILLED", rows_recovered=rows
                    )
                    stats["backfilled"] += 1
                    stats["rows_recovered"] += rows
                    metrics.record_gap_backfilled(table_name, rows)
                    logger.info(
                        f"Backfilled {rows} rows for item {item_id} "
                        f"({gap_start} to {gap_end})"
                    )
                else:
                    update_gap_status(
                        conn,
                        gap_id,
                        "UNRECOVERABLE",
                        error_message="No data available from Wiki API for this window",
                    )
                    stats["unrecoverable"] += 1

                # Rate limit API calls
                time.sleep(0.5)

            except Exception as e:
                update_gap_status(
                    conn, gap_id, "DETECTED", error_message=str(e)
                )
                stats["errors"] += 1
                metrics.record_backfill_error(table_name, type(e).__name__)
        else:
            # Global gap - mark as detected only, need manual intervention
            logger.warning(f"Global gap detected: {gap_start} to {gap_end}")

    return stats


def get_unrecoverable_count(conn) -> int:
    """Get count of unrecoverable gaps."""
    query = """
        SELECT COUNT(*)
        FROM data_quality
        WHERE status = 'UNRECOVERABLE'
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            return result[0] if result else 0
    except psycopg2.Error:
        return 0


def run_gap_detection(conn, lookback_hours: int) -> dict:
    """Run a full gap detection cycle."""
    logger.info(f"Starting gap detection (lookback: {lookback_hours}h)")

    total_stats = {
        "items_checked": 0,
        "detected": 0,
        "backfilled": 0,
        "unrecoverable": 0,
        "errors": 0,
        "rows_recovered": 0,
    }

    # First check for global gaps (systemic failures)
    logger.info("Checking for global gaps...")
    global_gaps = detect_global_gaps(conn, lookback_hours)
    if global_gaps:
        logger.warning(f"Found {len(global_gaps)} global gaps")
        stats = process_gaps(conn, global_gaps)
        for key in stats:
            total_stats[key] += stats[key]

    # Get items to check
    items = get_tracked_items(conn, limit=BACKFILL_BATCH_SIZE)
    logger.info(f"Checking {len(items)} items for gaps")

    for i, item_id in enumerate(items):
        if shutdown_requested:
            logger.info("Shutdown requested, stopping gap detection")
            break

        gaps = detect_gaps_for_item(conn, item_id, lookback_hours)
        if gaps:
            logger.info(f"Item {item_id}: Found {len(gaps)} gaps")
            stats = process_gaps(conn, gaps, item_id)
            for key in stats:
                total_stats[key] += stats[key]

        total_stats["items_checked"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(items)} items checked")

    # Update unrecoverable count metric
    unrecoverable_count = get_unrecoverable_count(conn)
    metrics.set_unrecoverable_gaps("price_data_5min", unrecoverable_count)

    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Gap Detector Service")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (hourly) instead of one-shot",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_HOURS,
        help=f"Hours to look back for gaps (default: {LOOKBACK_HOURS})",
    )
    args = parser.parse_args()

    logger.info(f"Starting Gap Detector (lookback: {args.lookback}h)")

    # Add /health endpoint to Prometheus server
    add_health_routes_to_prometheus(health_checker)

    # Start Prometheus metrics server (now also serves /health)
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server started on port {METRICS_PORT} (also serves /health)")

    # Connect to database
    conn = get_db_connection()
    logger.info("Connected to PostgreSQL")

    # Register DB connection with health checker
    health_checker.set_db_connection(conn)

    if args.continuous:
        logger.info(f"Running in continuous mode (interval: {RUN_INTERVAL}s)")
        while not shutdown_requested:
            try:
                start_time = time.time()
                stats = run_gap_detection(conn, args.lookback)

                logger.info(
                    f"Gap detection complete: "
                    f"{stats['items_checked']} items, "
                    f"{stats['detected']} gaps detected, "
                    f"{stats['backfilled']} backfilled, "
                    f"{stats['rows_recovered']} rows recovered"
                )

                metrics.record_success()
                health_checker.record_collection()  # Update health checker

                # Sleep until next run
                elapsed = time.time() - start_time
                sleep_time = max(0, RUN_INTERVAL - elapsed)
                logger.info(f"Next run in {sleep_time:.0f}s")
                time.sleep(sleep_time)

            except psycopg2.OperationalError as e:
                logger.error(f"Database error: {e}")
                metrics.record_error("database")
                health_checker.set_db_connection(None)  # Mark DB as disconnected
                try:
                    conn = get_db_connection()
                    health_checker.set_db_connection(conn)  # Update health checker
                except Exception:
                    pass
                time.sleep(60)

            except Exception as e:
                logger.error(f"Gap detection error: {e}")
                metrics.record_error("unknown")
                time.sleep(60)
    else:
        # One-shot mode
        try:
            stats = run_gap_detection(conn, args.lookback)
            logger.info(
                f"Gap detection complete: "
                f"{stats['items_checked']} items checked, "
                f"{stats['detected']} gaps detected, "
                f"{stats['backfilled']} backfilled, "
                f"{stats['rows_recovered']} rows recovered"
            )
            metrics.record_success()
            health_checker.record_collection()  # Update health checker
        except Exception as e:
            logger.error(f"Gap detection failed: {e}")
            metrics.record_error("unknown")
            sys.exit(1)

    conn.close()
    logger.info("Gap Detector stopped")


if __name__ == "__main__":
    main()
