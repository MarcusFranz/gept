#!/usr/bin/env python3
"""
Populate buy_limit column in items table from OSRS Wiki API.

The OSRS Wiki maintains accurate buy limit data for all tradeable items.
This script fetches that data and updates our database.

Usage:
    python populate_buy_limits.py [--dry-run]

Options:
    --dry-run    Show what would be updated without making changes
"""

import argparse
import os
import sys
import time
from typing import Optional

import psycopg2
import requests

# OSRS Wiki API endpoint for item mapping (includes buy limits)
WIKI_API_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"

# Required headers per Wiki API guidelines
HEADERS = {
    "User-Agent": "GePT-BuyLimitUpdater/1.0 (https://gept.gg; contact@gept.gg)"
}


def get_db_connection():
    """Get database connection from environment."""
    # Try DATABASE_URL first (standard format)
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)

    # Fall back to individual env vars
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "osrs_data"),
        user=os.environ.get("DB_USER", "osrs_user"),
        password=os.environ.get("DB_PASS"),
    )


def fetch_wiki_buy_limits() -> dict[int, int]:
    """Fetch buy limits from OSRS Wiki API.

    Returns:
        Dict mapping item_id -> buy_limit
    """
    print("Fetching buy limits from OSRS Wiki API...")

    response = requests.get(WIKI_API_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    data = response.json()

    buy_limits = {}
    for item in data:
        item_id = item.get("id")
        limit = item.get("limit")

        if item_id is not None and limit is not None:
            buy_limits[item_id] = limit

    print(f"Fetched {len(buy_limits):,} items with buy limits")
    return buy_limits


def get_current_items(conn) -> set[int]:
    """Get all item IDs currently in database."""
    with conn.cursor() as cur:
        cur.execute("SELECT item_id FROM items")
        return {row[0] for row in cur.fetchall()}


def update_buy_limits(conn, buy_limits: dict[int, int], dry_run: bool = False) -> tuple[int, int]:
    """Update buy_limit column for items in database.

    Args:
        conn: Database connection
        buy_limits: Dict mapping item_id -> buy_limit
        dry_run: If True, don't commit changes

    Returns:
        Tuple of (updated_count, skipped_count)
    """
    current_items = get_current_items(conn)

    # Filter to only items we have in our database
    updates = {
        item_id: limit
        for item_id, limit in buy_limits.items()
        if item_id in current_items
    }

    skipped = len(buy_limits) - len(updates)

    if dry_run:
        print(f"\n[DRY RUN] Would update {len(updates):,} items, skip {skipped:,} (not in DB)")

        # Show sample of updates
        print("\nSample updates:")
        with conn.cursor() as cur:
            for item_id, limit in list(updates.items())[:10]:
                cur.execute("SELECT name FROM items WHERE item_id = %s", (item_id,))
                row = cur.fetchone()
                name = row[0] if row else "Unknown"
                print(f"  {name} (id={item_id}): buy_limit = {limit:,}")

        return len(updates), skipped

    print(f"\nUpdating {len(updates):,} items...")

    with conn.cursor() as cur:
        # Batch update for efficiency
        updated = 0
        batch_size = 500
        items = list(updates.items())

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Use executemany with UPDATE
            cur.executemany(
                "UPDATE items SET buy_limit = %s, updated_at = NOW() WHERE item_id = %s",
                [(limit, item_id) for item_id, limit in batch]
            )
            updated += len(batch)

            if updated % 1000 == 0:
                print(f"  Updated {updated:,} items...")

    conn.commit()
    print(f"Successfully updated {updated:,} items")

    return updated, skipped


def show_stats(conn):
    """Show current buy_limit statistics."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(buy_limit) as has_limit,
                COUNT(*) - COUNT(buy_limit) as missing_limit,
                MIN(buy_limit) as min_limit,
                MAX(buy_limit) as max_limit,
                ROUND(AVG(buy_limit)) as avg_limit
            FROM items
        """)
        row = cur.fetchone()

        print("\nDatabase statistics:")
        print(f"  Total items: {row[0]:,}")
        print(f"  With buy_limit: {row[1]:,}")
        print(f"  Missing buy_limit: {row[2]:,}")
        if row[1] > 0:
            print(f"  Min limit: {row[3]:,}")
            print(f"  Max limit: {row[4]:,}")
            print(f"  Avg limit: {row[5]:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Populate buy limits from OSRS Wiki")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    args = parser.parse_args()

    # Check for required env var
    if not os.environ.get("DB_PASS") and not os.environ.get("DATABASE_URL"):
        print("Error: DB_PASS or DATABASE_URL environment variable required")
        print("Run: source /home/ubuntu/gept/.env")
        sys.exit(1)

    try:
        conn = get_db_connection()

        print("=== Buy Limit Updater ===\n")

        # Show current state
        print("Before update:")
        show_stats(conn)

        # Fetch from Wiki
        buy_limits = fetch_wiki_buy_limits()

        # Update database
        updated, skipped = update_buy_limits(conn, buy_limits, dry_run=args.dry_run)

        # Show new state
        if not args.dry_run:
            print("\nAfter update:")
            show_stats(conn)

        conn.close()

    except requests.RequestException as e:
        print(f"Error fetching from Wiki API: {e}")
        sys.exit(1)
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
