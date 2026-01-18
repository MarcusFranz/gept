#!/usr/bin/env python3
"""
Player Count Collector - PostgreSQL Version

Collects OSRS player counts from the homepage every 60 seconds
and stores them in PostgreSQL.

Environment Variables:
    DB_HOST: PostgreSQL host (default: localhost)
    DB_PORT: PostgreSQL port (default: 5432)
    DB_NAME: Database name (default: osrs_data)
    DB_USER: Database user (default: osrs_user)
    DB_PASS: Database password
    METRICS_PORT: Prometheus metrics port (default: 9104)
"""

import os
import re
import sys
import time
import signal
import logging
from datetime import datetime

import requests
import psycopg2
from psycopg2.extras import execute_values

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from shared.metrics import get_data_quality_metrics

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'dbname': os.getenv('DB_NAME', 'osrs_data'),
    'user': os.getenv('DB_USER', 'osrs_user'),
    'password': os.environ["DB_PASS"]
}

URL = "https://oldschool.runescape.com/"
USER_AGENT = "GePT-Data-Collector/3.0 - Contact: @marcusfranz (Discord)"
FETCH_INTERVAL_SECONDS = int(os.getenv('COLLECTION_INTERVAL', '60'))
METRICS_PORT = int(os.getenv('METRICS_PORT', '9104'))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
if PROMETHEUS_AVAILABLE:
    REQUESTS_TOTAL = Counter(
        'gept_player_count_requests_total',
        'Total requests to OSRS homepage',
        ['status']
    )
    PLAYER_COUNT = Gauge(
        'gept_player_count_current',
        'Current OSRS player count'
    )
    COLLECTION_DURATION = Histogram(
        'gept_player_count_duration_seconds',
        'Time to collect player count'
    )
    LAST_COLLECTION = Gauge(
        'gept_player_count_last_timestamp',
        'Timestamp of last successful collection'
    )

# Data quality metrics
dq_metrics = get_data_quality_metrics('player_count')


class PlayerCountCollector:
    def __init__(self):
        self.running = True
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _ensure_connection(self):
        """Reconnect if connection is lost."""
        try:
            if self.conn is None or self.conn.closed:
                self._connect()
            else:
                # Test connection
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            self._connect()

    def fetch_player_count(self):
        """Fetch OSRS player count from the homepage."""
        start_time = time.time()

        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(URL, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.error(f"Failed to fetch homepage: {response.status_code}")
                if PROMETHEUS_AVAILABLE:
                    REQUESTS_TOTAL.labels(status='error').inc()
                return

            # Parse player count from HTML
            content = response.text
            match = re.search(r"There are currently ([\d,]+) people playing!", content)

            if match:
                count_str = match.group(1).replace(",", "")
                count = int(count_str)

                # Insert into PostgreSQL
                self._ensure_connection()
                try:
                    with self.conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO player_counts (timestamp, count, fetched_at)
                            VALUES (NOW(), %s, NOW())
                            ON CONFLICT (timestamp) DO UPDATE SET count = EXCLUDED.count
                        """, (count,))
                except psycopg2.Error:
                    dq_metrics.record_db_commit_failure()
                    raise

                logger.info(f"Logged player count: {count:,}")

                if PROMETHEUS_AVAILABLE:
                    REQUESTS_TOTAL.labels(status='success').inc()
                    PLAYER_COUNT.set(count)
                    LAST_COLLECTION.set(time.time())
            else:
                logger.warning("Could not find player count pattern in HTML")
                dq_metrics.record_api_validation_failure('pattern_not_found')
                if PROMETHEUS_AVAILABLE:
                    REQUESTS_TOTAL.labels(status='parse_error').inc()

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            if PROMETHEUS_AVAILABLE:
                REQUESTS_TOTAL.labels(status='error').inc()
        except Exception as e:
            logger.error(f"Error in fetch_player_count: {e}")
            if PROMETHEUS_AVAILABLE:
                REQUESTS_TOTAL.labels(status='error').inc()
        finally:
            if PROMETHEUS_AVAILABLE:
                COLLECTION_DURATION.observe(time.time() - start_time)

    def run(self):
        """Main collection loop."""
        logger.info(f"Starting Player Count Collector (interval: {FETCH_INTERVAL_SECONDS}s)")

        if PROMETHEUS_AVAILABLE:
            start_http_server(METRICS_PORT)
            logger.info(f"Prometheus metrics available on port {METRICS_PORT}")

        while self.running:
            self.fetch_player_count()
            time.sleep(FETCH_INTERVAL_SECONDS)

    def shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
        if self.conn:
            self.conn.close()
        sys.exit(0)


if __name__ == "__main__":
    collector = PlayerCountCollector()
    signal.signal(signal.SIGINT, collector.shutdown)
    signal.signal(signal.SIGTERM, collector.shutdown)
    collector.run()
