#!/usr/bin/env python3
"""
Item Metadata Updater - PostgreSQL Version

Fetches item metadata from the RuneScape Wiki API and updates
the PostgreSQL items table. Runs daily.

Environment Variables:
    DB_HOST: PostgreSQL host (default: localhost)
    DB_PORT: PostgreSQL port (default: 5432)
    DB_NAME: Database name (default: osrs_data)
    DB_USER: Database user (default: osrs_user)
    DB_PASS: Database password
    METRICS_PORT: Prometheus metrics port (default: 9105)
"""

import os
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

MAPPING_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
USER_AGENT = "GePT-Data-Collector/3.0 - Contact: @marcusfranz (Discord)"
FETCH_INTERVAL_SECONDS = int(os.getenv('COLLECTION_INTERVAL', '86400'))  # 24 hours
METRICS_PORT = int(os.getenv('METRICS_PORT', '9105'))

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
        'gept_items_requests_total',
        'Total requests to Wiki API',
        ['status']
    )
    ITEMS_UPDATED = Counter(
        'gept_items_updated_total',
        'Total items updated'
    )
    COLLECTION_DURATION = Histogram(
        'gept_items_duration_seconds',
        'Time to update items'
    )
    LAST_COLLECTION = Gauge(
        'gept_items_last_timestamp',
        'Timestamp of last successful update'
    )

# Data quality metrics
dq_metrics = get_data_quality_metrics('items')


class ItemUpdater:
    def __init__(self):
        self.running = True
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
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
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            self._connect()

    def fetch_and_update_items(self):
        """Fetch item mapping from Wiki API and update PostgreSQL."""
        start_time = time.time()

        try:
            logger.info("Fetching item mapping from Wiki API...")
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(MAPPING_URL, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.error(f"Failed to fetch mapping: {response.status_code}")
                if PROMETHEUS_AVAILABLE:
                    REQUESTS_TOTAL.labels(status='error').inc()
                return

            data = response.json()
            logger.info(f"Received {len(data)} items from API")

            # Prepare data for upsert
            # Map API fields to database columns
            items_data = []
            null_id_count = 0
            null_name_count = 0
            for item in data:
                item_id = item.get('id')
                name = item.get('name')

                # Track missing required fields
                if item_id is None:
                    null_id_count += 1
                    continue  # Skip items without ID
                if not name:
                    null_name_count += 1

                items_data.append((
                    item_id,
                    name,
                    item.get('examine'),
                    item.get('members', False),
                    True,  # tradeable - all items from this API are tradeable
                    item.get('lowalch'),
                    item.get('highalch'),
                    item.get('value'),  # cost
                    item.get('limit'),  # GE buy limit
                    f"https://oldschool.runescape.wiki/w/{(name or '').replace(' ', '_')}"
                ))

            # Record null value metrics
            if null_id_count > 0:
                dq_metrics.record_null_values('id', null_id_count)
            if null_name_count > 0:
                dq_metrics.record_null_values('name', null_name_count)

            # Upsert into PostgreSQL
            self._ensure_connection()
            try:
                with self.conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO items (item_id, name, examine, members, tradeable, lowalch, highalch, cost, buy_limit, wiki_url, updated_at)
                        VALUES %s
                        ON CONFLICT (item_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            examine = EXCLUDED.examine,
                            members = EXCLUDED.members,
                            tradeable = EXCLUDED.tradeable,
                            lowalch = EXCLUDED.lowalch,
                            highalch = EXCLUDED.highalch,
                            cost = EXCLUDED.cost,
                            buy_limit = EXCLUDED.buy_limit,
                            wiki_url = EXCLUDED.wiki_url,
                            updated_at = NOW()
                        """,
                        items_data,
                        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"
                    )
                    self.conn.commit()
            except psycopg2.Error:
                dq_metrics.record_db_commit_failure()
                raise

            logger.info(f"Updated {len(items_data)} items in PostgreSQL")

            if PROMETHEUS_AVAILABLE:
                REQUESTS_TOTAL.labels(status='success').inc()
                ITEMS_UPDATED.inc(len(items_data))
                LAST_COLLECTION.set(time.time())

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            if PROMETHEUS_AVAILABLE:
                REQUESTS_TOTAL.labels(status='error').inc()
        except Exception as e:
            logger.error(f"Error in fetch_and_update_items: {e}")
            if PROMETHEUS_AVAILABLE:
                REQUESTS_TOTAL.labels(status='error').inc()
        finally:
            if PROMETHEUS_AVAILABLE:
                COLLECTION_DURATION.observe(time.time() - start_time)

    def run(self):
        """Main update loop."""
        logger.info(f"Starting Item Updater (interval: {FETCH_INTERVAL_SECONDS}s)")

        if PROMETHEUS_AVAILABLE:
            start_http_server(METRICS_PORT)
            logger.info(f"Prometheus metrics available on port {METRICS_PORT}")

        while self.running:
            self.fetch_and_update_items()
            time.sleep(FETCH_INTERVAL_SECONDS)

    def shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
        if self.conn:
            self.conn.close()
        sys.exit(0)


if __name__ == "__main__":
    updater = ItemUpdater()
    signal.signal(signal.SIGINT, updater.shutdown)
    signal.signal(signal.SIGTERM, updater.shutdown)
    updater.run()
