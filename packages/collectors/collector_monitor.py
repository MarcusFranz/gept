#!/usr/bin/env python3
"""Collector Monitor - Data Freshness Monitoring Service.

Central monitoring service that continuously checks data freshness across all
critical tables and exposes Prometheus metrics for alerting.

Runs as a Docker container (port 9106) and checks data freshness every 60 seconds.
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import psycopg2
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

METRICS_PORT = int(os.getenv("METRICS_PORT", "9106"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))

# Data freshness thresholds (in seconds)
# Alert if data is older than these thresholds
FRESHNESS_THRESHOLDS = {
    "price_data_5min": 600,      # 10 minutes (2x the collection interval)
    "prices_latest_1m": 120,     # 2 minutes (2x the collection interval)
    "prices_1h": 7200,           # 2 hours (2x the collection interval)
    "player_counts": 300,        # 5 minutes (5x the collection interval)
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize metrics
metrics = MonitorMetrics("monitor")

# Health checker for /health endpoint
health_checker = HealthChecker(
    service_name='collector-monitor',
    collection_interval=CHECK_INTERVAL,
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


def check_table_freshness(conn, table_name: str, timestamp_column: str = "timestamp") -> float:
    """
    Check the age of the most recent data in a table.

    Args:
        conn: Database connection
        table_name: Name of the table to check
        timestamp_column: Name of the timestamp column

    Returns:
        Age in seconds of the most recent row, or infinity if table is empty
    """
    query = f"""
        SELECT EXTRACT(EPOCH FROM (NOW() - MAX({timestamp_column})))
        FROM {table_name}
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            if result and result[0] is not None:
                return float(result[0])
            # Table is empty
            return float("inf")
    except psycopg2.Error as e:
        logger.error(f"Error checking freshness for {table_name}: {e}")
        return float("inf")


def check_all_tables(conn) -> dict:
    """
    Check freshness of all monitored tables.

    Returns:
        Dictionary mapping table name to age in seconds
    """
    results = {}

    for table_name, threshold in FRESHNESS_THRESHOLDS.items():
        age = check_table_freshness(conn, table_name)
        results[table_name] = {
            "age_seconds": age,
            "threshold_seconds": threshold,
            "is_fresh": age <= threshold,
        }

        # Record metrics
        metrics.record_data_age(table_name, age, threshold)

        # Log status
        status = "OK" if age <= threshold else "STALE"
        if age == float("inf"):
            logger.warning(f"[{status}] {table_name}: No data found")
        else:
            logger.info(
                f"[{status}] {table_name}: {age:.0f}s old (threshold: {threshold}s)"
            )

    return results


def get_table_stats(conn) -> dict:
    """Get additional statistics about monitored tables."""
    stats = {}

    for table_name in FRESHNESS_THRESHOLDS:
        try:
            with conn.cursor() as cur:
                # Get row count (approximate for large tables)
                cur.execute(
                    f"""
                    SELECT reltuples::BIGINT
                    FROM pg_class
                    WHERE relname = %s
                """,
                    (table_name,),
                )
                result = cur.fetchone()
                row_count = result[0] if result else 0

                stats[table_name] = {
                    "approximate_rows": row_count,
                }
        except psycopg2.Error as e:
            logger.error(f"Error getting stats for {table_name}: {e}")
            stats[table_name] = {"error": str(e)}

    return stats


def main():
    logger.info(
        f"Starting Collector Monitor (port={METRICS_PORT}, interval={CHECK_INTERVAL}s)"
    )
    logger.info(f"Monitoring tables: {list(FRESHNESS_THRESHOLDS.keys())}")

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

    # Initialize threshold metrics (static values)
    for table_name, threshold in FRESHNESS_THRESHOLDS.items():
        metrics.freshness_threshold.labels(table=table_name).set(threshold)

    # Main monitoring loop
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not shutdown_requested:
        try:
            start_time = time.time()

            # Check all tables
            results = check_all_tables(conn)

            # Count stale tables
            stale_count = sum(1 for r in results.values() if not r["is_fresh"])

            # Update health status
            if stale_count == 0:
                metrics.set_healthy()
            else:
                metrics.set_unhealthy()
                logger.warning(f"{stale_count} table(s) have stale data")

            # Record successful check
            metrics.record_success()
            health_checker.record_collection()  # Update health checker
            consecutive_errors = 0

            duration = time.time() - start_time
            logger.info(f"Check completed in {duration:.2f}s")

        except psycopg2.OperationalError as e:
            logger.error(f"Database connection error: {e}")
            consecutive_errors += 1
            metrics.record_error("database_connection")
            health_checker.set_db_connection(None)  # Mark DB as disconnected

            # Try to reconnect
            try:
                conn = get_db_connection()
                health_checker.set_db_connection(conn)  # Update health checker
                logger.info("Reconnected to PostgreSQL")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect: {reconnect_error}")

        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            consecutive_errors += 1
            metrics.record_error("unknown")

        # Check if we should mark service as unhealthy
        if consecutive_errors >= max_consecutive_errors:
            metrics.set_unhealthy()
            logger.error(
                f"Service unhealthy: {consecutive_errors} consecutive errors"
            )

        # Sleep until next check
        elapsed = time.time() - start_time
        sleep_time = max(0, CHECK_INTERVAL - elapsed)
        time.sleep(sleep_time)

    # Cleanup
    conn.close()
    logger.info("Collector Monitor stopped")


if __name__ == "__main__":
    main()
