#!/usr/bin/env python3
"""
Backfill 5-minute price data from OSRS Wiki API.

Uses the /5m endpoint which returns ALL items for a specific 5-minute window.
This endpoint has historical data going back ~6+ months, unlike the /timeseries
endpoint which only has ~2 days of 5m data.

Usage:
    python backfill_5min_data.py [--dry-run] [--item-id ID] [--min-price PRICE] [--limit N]

Options:
    --dry-run       Show what would be fetched without inserting
    --item-id ID    Backfill a specific item only (or comma-separated list)
    --min-price     Minimum average price to consider (default: 1000000)
    --limit N       Maximum number of items to backfill (default: 100)
    --days N        Number of days of history to fetch (default: 180)
    --interval N    Minutes between samples (default: 5, use 60 for hourly)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
import requests

# OSRS Wiki API endpoint for 5-minute snapshots
WIKI_API_URL = "https://prices.runescape.wiki/api/v1/osrs/5m"

# Required headers per Wiki API guidelines
HEADERS = {
    "User-Agent": "GePT-Backfill/1.0 (https://gept.gg; contact@gept.gg)"
}

# Rate limiting - Wiki API asks for max 1 request per second
REQUEST_DELAY = 1.2  # seconds between requests


def get_db_connection():
    """Get database connection from environment."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "osrs_data"),
        user=os.environ.get("DB_USER", "osrs_user"),
        password=os.environ.get("DB_PASS"),
    )


def get_items_needing_backfill(conn, min_price: int = 1_000_000,
                                min_days_needed: int = 90,
                                limit: int = 100) -> list:
    """Find items that need historical data backfilled.

    Returns items that:
    - Have average price > min_price
    - Have less than min_days_needed of 5-min data
    """
    with conn.cursor() as cur:
        cur.execute("""
            WITH item_coverage AS (
                SELECT
                    p.item_id,
                    i.name,
                    COUNT(*) as row_count,
                    MIN(p.timestamp) as first_date,
                    MAX(p.timestamp) as last_date,
                    EXTRACT(EPOCH FROM (MAX(p.timestamp) - MIN(p.timestamp))) / 86400 as days_covered,
                    AVG(p.avg_high_price) FILTER (
                        WHERE p.avg_high_price > 0
                        AND p.avg_high_price != 'NaN'::float
                    ) as avg_price
                FROM price_data_5min p
                JOIN items i ON p.item_id = i.item_id
                WHERE i.tradeable = true
                GROUP BY p.item_id, i.name
            )
            SELECT item_id, name, row_count, days_covered, avg_price, first_date
            FROM item_coverage
            WHERE avg_price > %s
              AND days_covered < %s
            ORDER BY avg_price DESC
            LIMIT %s
        """, (min_price, min_days_needed, limit))

        return [
            {
                'item_id': row[0],
                'name': row[1],
                'row_count': row[2],
                'days_covered': row[3],
                'avg_price': row[4],
                'first_date': row[5]
            }
            for row in cur.fetchall()
        ]


def get_item_info(conn, item_ids: list) -> dict:
    """Get item info for a list of item IDs."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT item_id, name FROM items
            WHERE item_id = ANY(%s)
        """, (item_ids,))
        return {row[0]: row[1] for row in cur.fetchall()}


def fetch_5m_snapshot(timestamp: int) -> Optional[dict]:
    """Fetch 5-minute price snapshot from Wiki API.

    Args:
        timestamp: Unix timestamp (must be divisible by 300)

    Returns:
        Dict of item_id -> price data, or None on error
    """
    # Align timestamp to 5-minute boundary
    timestamp = timestamp - (timestamp % 300)

    params = {"timestamp": timestamp}

    try:
        response = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'error' in data:
            print(f"  API error: {data['error']}")
            return None

        return data.get('data', {})

    except requests.RequestException as e:
        print(f"  Error fetching timestamp {timestamp}: {e}")
        return None


def get_existing_timestamps(conn, item_ids: list, start_ts: datetime, end_ts: datetime) -> dict:
    """Get existing timestamps for items in the given time range.

    Returns:
        Dict of item_id -> set of timestamps
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT item_id, timestamp FROM price_data_5min
            WHERE item_id = ANY(%s)
              AND timestamp >= %s
              AND timestamp <= %s
        """, (item_ids, start_ts, end_ts))

        result = {}
        for row in cur.fetchall():
            item_id, ts = row
            if item_id not in result:
                result[item_id] = set()
            result[item_id].add(ts)

        return result


def backfill_items(conn, item_ids: list, item_names: dict,
                   days: int = 180, interval_minutes: int = 5,
                   dry_run: bool = False, all_items: bool = False) -> dict:
    """Backfill historical data for a list of items.

    Args:
        conn: Database connection
        item_ids: List of item IDs to backfill
        item_names: Dict mapping item_id -> name
        days: Number of days of history to fetch
        interval_minutes: Minutes between samples (5, 15, 30, 60, etc.)
        dry_run: If True, don't actually insert

    Returns:
        Dict of results: {item_id: {'inserted': n, 'skipped': n}}
    """
    results = {item_id: {'inserted': 0, 'skipped': 0} for item_id in item_ids}

    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    # Align to 5-minute boundaries
    end_ts = int(end_time.timestamp())
    end_ts = end_ts - (end_ts % 300)
    start_ts = int(start_time.timestamp())
    start_ts = start_ts - (start_ts % 300)

    # Calculate number of intervals
    interval_seconds = interval_minutes * 60
    total_intervals = (end_ts - start_ts) // interval_seconds

    if all_items:
        print(f"\nBackfilling ALL items from API responses")
    else:
        print(f"\nBackfilling {len(item_ids)} items:")
        for item_id in item_ids:
            name = item_names.get(item_id, f"Item {item_id}")
            print(f"  - {name} (ID: {item_id})")
    print(f"\nTime range: {start_time.date()} to {end_time.date()} ({days} days)")
    print(f"Sampling interval: every {interval_minutes} minutes")
    print(f"Total API requests needed: {total_intervals:,}")

    # Get existing timestamps to avoid duplicates (skip for all_items mode - too many to check)
    existing = {}
    if not all_items:
        print("\nChecking existing data...")
        existing = get_existing_timestamps(
            conn, item_ids,
            datetime.fromtimestamp(start_ts, timezone.utc),
            datetime.fromtimestamp(end_ts, timezone.utc)
        )
        for item_id in item_ids:
            count = len(existing.get(item_id, set()))
            print(f"  {item_names.get(item_id, item_id)}: {count:,} existing records")
    else:
        print("\nSkipping existing data check (all-items mode uses ON CONFLICT)")

    if dry_run:
        print("\n[DRY RUN] Would fetch and insert data for these items")
        return results

    # Iterate through time, newest to oldest
    current_ts = end_ts
    requests_made = 0
    last_progress = 0

    print("\nFetching data...")

    with conn.cursor() as cur:
        while current_ts >= start_ts:
            # Fetch snapshot
            data = fetch_5m_snapshot(current_ts)

            if data:
                ts_datetime = datetime.fromtimestamp(current_ts, timezone.utc)

                # Process items - all items from API or just specified ones
                items_to_process = data.keys() if all_items else [str(i) for i in item_ids]

                for item_key in items_to_process:
                    try:
                        item_id = int(item_key)
                    except ValueError:
                        continue

                    # Initialize result tracking for new items
                    if item_id not in results:
                        results[item_id] = {'inserted': 0, 'skipped': 0}

                    # Skip if we already have this data point
                    if item_id in existing and ts_datetime in existing[item_id]:
                        results[item_id]['skipped'] += 1
                        continue

                    # Get item data from snapshot
                    item_data = data.get(item_key)
                    if not item_data:
                        continue

                    avg_high = item_data.get('avgHighPrice')
                    avg_low = item_data.get('avgLowPrice')
                    high_vol = item_data.get('highPriceVolume', 0)
                    low_vol = item_data.get('lowPriceVolume', 0)

                    # Skip if no price data
                    if avg_high is None and avg_low is None:
                        continue

                    try:
                        cur.execute("""
                            INSERT INTO price_data_5min
                            (item_id, timestamp, avg_high_price, avg_low_price,
                             high_price_volume, low_price_volume)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (item_id, timestamp) DO NOTHING
                        """, (item_id, ts_datetime, avg_high, avg_low, high_vol, low_vol))
                        results[item_id]['inserted'] += 1
                    except Exception as e:
                        print(f"  Error inserting: {e}")

            requests_made += 1

            # Progress indicator
            progress = int(100 * requests_made / total_intervals)
            if progress >= last_progress + 5:
                conn.commit()  # Periodic commit
                ts_date = datetime.fromtimestamp(current_ts, timezone.utc).date()
                total_inserted = sum(r['inserted'] for r in results.values())
                print(f"  {progress}% complete - at {ts_date}, {total_inserted:,} records inserted")
                last_progress = progress

            # Rate limiting
            time.sleep(REQUEST_DELAY)

            # Move to next interval
            current_ts -= interval_seconds

    conn.commit()

    total_inserted = sum(r['inserted'] for r in results.values())
    total_skipped = sum(r['skipped'] for r in results.values())
    print(f"\nResults: {total_inserted:,} inserted, {total_skipped:,} skipped across {len(results)} items")

    if not all_items:
        # Show per-item results only for targeted backfills
        for item_id in item_ids:
            name = item_names.get(item_id, f"Item {item_id}")
            r = results.get(item_id, {'inserted': 0, 'skipped': 0})
            print(f"  {name}: {r['inserted']:,} inserted, {r['skipped']:,} skipped")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Backfill 5-minute price data from OSRS Wiki API"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without inserting")
    parser.add_argument("--item-id", type=str,
                        help="Backfill specific item(s) - ID or comma-separated list")
    parser.add_argument("--min-price", type=int, default=1_000_000,
                        help="Minimum average price to consider (default: 1M)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum items to backfill (default: 100)")
    parser.add_argument("--days", type=int, default=180,
                        help="Days of history to fetch (default: 180)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between samples (default: 5, use 60 for hourly)")
    parser.add_argument("--all-items", action="store_true",
                        help="Store ALL items from each API response (not just specified ones)")
    args = parser.parse_args()

    # Check for required env var
    if not os.environ.get("DB_PASS") and not os.environ.get("DATABASE_URL"):
        print("Error: DB_PASS or DATABASE_URL environment variable required")
        print("Run: source /home/ubuntu/gept/.env")
        sys.exit(1)

    print("=" * 70)
    print("OSRS 5-MINUTE DATA BACKFILL")
    print("=" * 70)

    if args.dry_run:
        print("DRY RUN MODE - No data will be inserted\n")

    if args.all_items:
        print("ALL ITEMS MODE - Storing all items from each API response\n")

    conn = get_db_connection()

    try:
        if args.all_items:
            # Store ALL items from API responses
            backfill_items(
                conn, [], {},
                days=args.days,
                interval_minutes=args.interval,
                dry_run=args.dry_run,
                all_items=True
            )
        elif args.item_id:
            # Parse item IDs
            item_ids = [int(x.strip()) for x in args.item_id.split(',')]
            item_names = get_item_info(conn, item_ids)

            # Check all items exist
            missing = [i for i in item_ids if i not in item_names]
            if missing:
                print(f"Error: Items not found in database: {missing}")
                sys.exit(1)

            backfill_items(
                conn, item_ids, item_names,
                days=args.days,
                interval_minutes=args.interval,
                dry_run=args.dry_run
            )
        else:
            # Find items needing backfill
            print(f"\nFinding items with avg price > {args.min_price:,} GP and < 90 days of data...")
            items = get_items_needing_backfill(
                conn,
                min_price=args.min_price,
                min_days_needed=90,
                limit=args.limit
            )

            if not items:
                print("No items found needing backfill!")
                return

            print(f"Found {len(items)} items to backfill:\n")
            for item in items:
                print(f"  {item['name']}: {item['days_covered']:.1f} days, "
                      f"avg {item['avg_price']:,.0f} GP")

            item_ids = [i['item_id'] for i in items]
            item_names = {i['item_id']: i['name'] for i in items}

            backfill_items(
                conn, item_ids, item_names,
                days=args.days,
                interval_minutes=args.interval,
                dry_run=args.dry_run
            )

        print("\n" + "=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
