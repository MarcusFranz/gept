#!/usr/bin/env python3
"""News Collector - PostgreSQL version.

Collects OSRS news from RSS feeds and wiki changes.
Stores to PostgreSQL osrs_news table.
"""

import logging
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import psycopg2
from psycopg2.extras import execute_values
from prometheus_client import Counter, Gauge, start_http_server
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

METRICS_PORT = int(os.getenv("METRICS_PORT", "9102"))
COLLECTION_INTERVAL = int(os.getenv("COLLECTION_INTERVAL", "1800"))  # 30 min
USER_AGENT = "GePT-NewsCollector/3.0 (PostgreSQL)"

# RSS Feeds
FEEDS = [
    ("https://secure.runescape.com/m=news/latest_news.rss?oldschool=true", "official"),
    ("https://oldschool.runescape.wiki/w/Special:RecentChanges?feed=rss&namespace=0&hidebots=1&hideminor=1", "wiki"),
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
ARTICLES_COLLECTED = Counter("gept_news_articles_total", "Total news articles collected")
LAST_COLLECTION = Gauge("gept_news_last_timestamp", "Last successful collection")
DB_RECONNECT_SUCCESS = Counter("gept_news_db_reconnect_success_total", "Successful DB reconnections")
DB_RECONNECT_FAILURES = Counter("gept_news_db_reconnect_failures_total", "Failed DB reconnection attempts")
API_RETRIES = Counter("gept_news_api_retries_total", "API request retry attempts")
CIRCUIT_OPEN = Counter("gept_news_circuit_open_total", "Times circuit breaker opened")
CIRCUIT_STATE = Gauge("gept_news_circuit_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)")
DATA_TRUNCATIONS = Counter("gept_news_data_truncations_total", "Total field truncations", ["field"])

# Field length limits (matching database schema)
FIELD_LIMITS = {
    "guid": 255,
    "title": 500,
    "link": 2000,
    "description": 10000,
    "category": 100,
}

# Data quality metrics
dq_metrics = get_data_quality_metrics('news')

# Circuit breaker for RSS feeds
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

rss_breaker = CircuitBreaker(
    fail_max=CIRCUIT_FAIL_MAX,
    reset_timeout=CIRCUIT_RESET_TIMEOUT,
    name="rss_feeds",
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


def _truncate_field(value: str, field_name: str, limit: int) -> str:
    """Truncate field value and log/track if truncation occurred."""
    if value and len(value) > limit:
        original_len = len(value)
        truncated = value[:limit]
        DATA_TRUNCATIONS.labels(field=field_name).inc()
        logger.debug(f"Truncated {field_name}: {original_len} -> {limit} chars")
        return truncated
    return value or ""


def _validate_url(url: str) -> str:
    """Validate URL format, return empty string if invalid."""
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        logger.warning(f"Invalid URL (not http/https): {url[:50]}...")
        dq_metrics.record_validation_error('invalid_url_scheme')
        return ""
    # Check for reasonable URL length
    if len(url) > FIELD_LIMITS["link"]:
        logger.warning(f"URL too long ({len(url)} chars), truncating")
        DATA_TRUNCATIONS.labels(field="link").inc()
        return url[:FIELD_LIMITS["link"]]
    return url


def _log_retry(retry_state):
    """Log retry attempts and increment metrics."""
    API_RETRIES.inc()
    logger.warning(f"RSS fetch failed, retrying (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
    before_sleep=_log_retry,
    reraise=True
)
def _fetch_rss_with_retry(url: str) -> str:
    """Fetch RSS feed with retry logic."""
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.text

def fetch_rss(url: str) -> list[dict]:
    """Fetch and parse RSS feed with circuit breaker protection."""
    try:
        text = rss_breaker.call(_fetch_rss_with_retry, url)
        root = ET.fromstring(text)
        items = []

        for item in root.findall(".//item"):
            guid = item.findtext("guid") or item.findtext("link") or ""
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            description = item.findtext("description") or ""
            category = item.findtext("category") or ""
            pub_date_str = item.findtext("pubDate")

            # Track missing required fields
            if not guid:
                dq_metrics.record_null_value('guid')
            if not title:
                dq_metrics.record_null_value('title')

            try:
                pub_date = parsedate_to_datetime(pub_date_str) if pub_date_str else None
            except Exception:
                pub_date = None
                dq_metrics.record_validation_error('invalid_date_format')

            if not pub_date_str:
                dq_metrics.record_null_value('pub_date')

            # Sanitize and truncate fields with logging (Issue #79)
            sanitized_guid = _truncate_field(guid, "guid", FIELD_LIMITS["guid"])
            sanitized_title = _truncate_field(title, "title", FIELD_LIMITS["title"])
            sanitized_link = _validate_url(link)
            sanitized_description = _truncate_field(description, "description", FIELD_LIMITS["description"])
            sanitized_category = _truncate_field(category, "category", FIELD_LIMITS["category"])

            items.append({
                "guid": sanitized_guid,
                "title": sanitized_title,
                "link": sanitized_link,
                "description": sanitized_description,
                "category": sanitized_category,
                "pub_date": pub_date,
            })

        return items
    except CircuitBreakerError:
        logger.warning("Circuit breaker is open, skipping RSS fetch")
        return []
    except ET.ParseError as e:
        logger.error(f"XML parsing error for {url}: {e}")
        dq_metrics.record_api_validation_failure('xml_parse_error')
        return []
    except Exception as e:
        logger.error(f"Failed to fetch {url} after retries: {e}")
        return []


def store_news(conn, articles: list[dict]) -> int:
    """Store news articles, returning count of new rows."""
    if not articles:
        return 0

    rows = [
        (a["guid"], a["title"], a["link"], a["description"], a["category"], a["pub_date"])
        for a in articles
    ]

    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO osrs_news (guid, title, link, description, category, pub_date)
                VALUES %s
                ON CONFLICT (guid) DO NOTHING
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

    ARTICLES_COLLECTED.inc(inserted)
    return inserted


def run_collection(conn) -> int:
    """Run one collection cycle."""
    total_inserted = 0
    
    for url, feed_type in FEEDS:
        logger.info(f"Fetching {feed_type} feed...")
        articles = fetch_rss(url)
        
        # Add category prefix for wiki items
        if feed_type == "wiki":
            for a in articles:
                if not a["category"]:
                    a["category"] = "wiki_edit"
        
        inserted = store_news(conn, articles)
        total_inserted += inserted
        logger.info(f"  {feed_type}: {len(articles)} fetched, {inserted} new")
        
        time.sleep(1)  # Rate limit between feeds
    
    return total_inserted


def main():
    logger.info(f"Starting news PostgreSQL collector (interval={COLLECTION_INTERVAL}s)")
    
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server on port {METRICS_PORT}")
    
    conn = get_db_connection()
    logger.info("Connected to PostgreSQL")
    
    while not shutdown_requested:
        try:
            start = time.time()
            
            inserted = run_collection(conn)
            
            LAST_COLLECTION.set(time.time())
            logger.info(f"Collection complete: {inserted} new articles")
            
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
