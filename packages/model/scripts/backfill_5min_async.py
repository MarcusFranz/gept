#!/usr/bin/env python3
"""
Async backfill for 5-minute price data from OSRS Wiki API.

Uses asyncio to overlap API fetching with database insertion:
- Fetches next batch while processing current batch
- Maintains 1 request/second rate limit
- Auto-resumes from oldest timestamp in database

Usage:
    python backfill_5min_async.py [--target-date YYYY-MM-DD] [--workers N]
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
import queue
import threading

import aiohttp
import psycopg2
from psycopg2.extras import execute_values

# OSRS Wiki API endpoint
WIKI_API_URL = "https://prices.runescape.wiki/api/v1/osrs/5m"

# Required headers per Wiki API guidelines
HEADERS = {
    "User-Agent": "GePT-Backfill-Async/2.0 (https://gept.gg; contact@gept.gg)"
}

# Rate limit: 1 request per second (Wiki API guideline)
MIN_REQUEST_INTERVAL = 1.0


def get_db_connection():
    """Get database connection from environment."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "osrs_data"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASS", ""),
    )


def get_oldest_timestamp(conn) -> Optional[datetime]:
    """Get the oldest timestamp in the database to resume from."""
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(timestamp) FROM price_data_5min")
        result = cur.fetchone()[0]
        return result


def insert_batch(conn, rows: list) -> int:
    """Insert a batch of rows using execute_values for speed."""
    if not rows:
        return 0

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO price_data_5min
            (item_id, timestamp, avg_high_price, avg_low_price, high_price_volume, low_price_volume)
            VALUES %s
            ON CONFLICT (item_id, timestamp) DO NOTHING
            """,
            rows,
            page_size=1000
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted


async def fetch_snapshot(session: aiohttp.ClientSession, timestamp: int) -> Optional[dict]:
    """Fetch 5-minute price snapshot from Wiki API."""
    # Align to 5-minute boundary
    timestamp = timestamp - (timestamp % 300)

    params = {"timestamp": timestamp}

    try:
        async with session.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=30) as response:
            if response.status != 200:
                print(f"  API error {response.status} for {timestamp}")
                return None

            data = await response.json()

            if 'error' in data:
                print(f"  API error: {data['error']}")
                return None

            return {
                'timestamp': timestamp,
                'data': data.get('data', {})
            }

    except Exception as e:
        print(f"  Error fetching {timestamp}: {e}")
        return None


def process_snapshot(snapshot: dict) -> list:
    """Convert API snapshot to database rows."""
    if not snapshot or not snapshot.get('data'):
        return []

    ts = datetime.fromtimestamp(snapshot['timestamp'], timezone.utc)
    rows = []

    for item_key, item_data in snapshot['data'].items():
        try:
            item_id = int(item_key)
        except ValueError:
            continue

        avg_high = item_data.get('avgHighPrice')
        avg_low = item_data.get('avgLowPrice')
        high_vol = item_data.get('highPriceVolume', 0)
        low_vol = item_data.get('lowPriceVolume', 0)

        # Skip if no price data
        if avg_high is None and avg_low is None:
            continue

        rows.append((item_id, ts, avg_high, avg_low, high_vol, low_vol))

    return rows


async def fetch_worker(
    session: aiohttp.ClientSession,
    timestamps: list,
    fetch_queue: asyncio.Queue,
    progress: dict
):
    """Worker that fetches from API at rate limit."""
    last_request = 0

    for ts in timestamps:
        # Rate limiting
        now = time.time()
        elapsed = now - last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)

        last_request = time.time()
        snapshot = await fetch_snapshot(session, ts)

        if snapshot:
            await fetch_queue.put(snapshot)
            progress['fetched'] += 1
        else:
            progress['errors'] += 1

    # Signal completion
    await fetch_queue.put(None)


async def insert_worker(
    conn,
    fetch_queue: asyncio.Queue,
    progress: dict
):
    """Worker that processes fetched data and inserts to DB."""
    batch = []
    batch_size = 50  # Commit every 50 API responses (~85k rows)

    while True:
        snapshot = await fetch_queue.get()

        if snapshot is None:
            # End signal - insert remaining
            if batch:
                rows = []
                for s in batch:
                    rows.extend(process_snapshot(s))
                inserted = insert_batch(conn, rows)
                progress['inserted'] += inserted
            break

        batch.append(snapshot)

        if len(batch) >= batch_size:
            rows = []
            for s in batch:
                rows.extend(process_snapshot(s))
            inserted = insert_batch(conn, rows)
            progress['inserted'] += inserted
            batch = []

            # Progress update
            pct = (progress['fetched'] / progress['total']) * 100
            ts_date = datetime.fromtimestamp(snapshot['timestamp'], timezone.utc).date()
            print(f"  {pct:.1f}% - at {ts_date}, {progress['inserted']:,} rows inserted, {progress['errors']} errors")


async def run_backfill(
    start_ts: int,
    end_ts: int,
    conn
):
    """Run the async backfill."""
    # Generate timestamps (newest to oldest)
    timestamps = []
    current = start_ts
    while current >= end_ts:
        timestamps.append(current)
        current -= 300  # 5 minutes

    total = len(timestamps)
    print(f"\nBackfilling {total:,} timestamps")
    print(f"From: {datetime.fromtimestamp(start_ts, timezone.utc)}")
    print(f"To: {datetime.fromtimestamp(end_ts, timezone.utc)}")

    progress = {
        'fetched': 0,
        'inserted': 0,
        'errors': 0,
        'total': total
    }

    # Queue for producer-consumer pattern
    fetch_queue = asyncio.Queue(maxsize=100)  # Buffer up to 100 responses

    async with aiohttp.ClientSession() as session:
        # Start fetch and insert workers
        fetch_task = asyncio.create_task(
            fetch_worker(session, timestamps, fetch_queue, progress)
        )
        insert_task = asyncio.create_task(
            insert_worker(conn, fetch_queue, progress)
        )

        # Wait for both to complete
        await asyncio.gather(fetch_task, insert_task)

    return progress


def main():
    parser = argparse.ArgumentParser(
        description="Async backfill 5-minute price data from OSRS Wiki API"
    )
    parser.add_argument("--target-date", type=str, default="2021-01-21",
                        help="Target end date YYYY-MM-DD (default: 2021-01-21)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without inserting")
    args = parser.parse_args()

    # Check for required env var
    if not os.environ.get("DB_PASS") and not os.environ.get("DATABASE_URL"):
        # Allow empty password for local trust auth
        if not os.environ.get("DB_HOST"):
            print("Warning: No DB_PASS set, assuming local trust auth")

    print("=" * 70)
    print("OSRS 5-MINUTE DATA BACKFILL (ASYNC)")
    print("=" * 70)

    conn = get_db_connection()

    try:
        # Get current oldest timestamp to resume from
        oldest = get_oldest_timestamp(conn)

        if oldest:
            # Resume from 5 minutes before oldest (to catch any gaps)
            start_ts = int(oldest.timestamp()) - 300
            print(f"\nResuming from: {oldest}")
        else:
            # Start from now
            start_ts = int(datetime.now(timezone.utc).timestamp())
            start_ts = start_ts - (start_ts % 300)  # Align to 5-min
            print(f"\nStarting fresh from: {datetime.fromtimestamp(start_ts, timezone.utc)}")

        # Target end date
        target_date = datetime.strptime(args.target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_ts = int(target_date.timestamp())

        print(f"Target end date: {target_date.date()}")

        if start_ts <= end_ts:
            print("\nAlready complete! Start is at or before target.")
            return

        if args.dry_run:
            total = (start_ts - end_ts) // 300
            print(f"\n[DRY RUN] Would fetch {total:,} timestamps")
            return

        # Run async backfill
        progress = asyncio.run(run_backfill(start_ts, end_ts, conn))

        print("\n" + "=" * 70)
        print("BACKFILL COMPLETE")
        print(f"Fetched: {progress['fetched']:,}")
        print(f"Inserted: {progress['inserted']:,}")
        print(f"Errors: {progress['errors']}")
        print("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
