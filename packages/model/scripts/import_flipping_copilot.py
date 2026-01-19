#!/usr/bin/env python3
"""Import Flipping Copilot CSV data into training_trades table."""

import csv
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values


def parse_timestamp(ts_str: str) -> datetime | None:
    """Parse ISO timestamp from Flipping Copilot CSV."""
    if not ts_str or ts_str.strip() == '':
        return None
    try:
        # Format: 2026-01-15T21:31:57Z
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except ValueError:
        return None


def parse_decimal(val: str) -> float | None:
    """Parse decimal value, returning None for empty strings."""
    if not val or val.strip() == '':
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_int(val: str) -> int | None:
    """Parse integer value, returning None for empty strings."""
    if not val or val.strip() == '':
        return None
    try:
        return int(val)
    except ValueError:
        return None


def load_item_mapping(conn) -> dict[str, int]:
    """Load item name to ID mapping from database."""
    cur = conn.cursor()
    cur.execute("SELECT name, item_id FROM items")
    mapping = {row[0].lower(): row[1] for row in cur.fetchall()}
    cur.close()
    return mapping


def check_price_data_coverage(conn, first_buy_time: datetime) -> bool:
    """Check if price_data_5min has coverage for the given timestamp."""
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM price_data_5min WHERE timestamp <= %s LIMIT 1",
        (first_buy_time,)
    )
    result = cur.fetchone() is not None
    cur.close()
    return result


def import_csv(csv_path: str, conn, price_data_start: datetime = None):
    """Import Flipping Copilot CSV into training_trades table.

    Args:
        csv_path: Path to the CSV file
        conn: Database connection
        price_data_start: Earliest timestamp with price data (for has_price_data flag)
    """
    # Load item name mapping
    item_mapping = load_item_mapping(conn)
    print(f"Loaded {len(item_mapping)} item mappings")

    # Get price data start time if not provided
    if price_data_start is None:
        cur = conn.cursor()
        cur.execute("SELECT MIN(timestamp) FROM price_data_5min")
        result = cur.fetchone()
        price_data_start = result[0] if result and result[0] else datetime(2025, 7, 21)
        cur.close()
        print(f"Price data starts at: {price_data_start}")

    # Parse CSV
    trades = []
    unmapped_items = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            first_buy_time = parse_timestamp(row['First buy time'])
            if first_buy_time is None:
                continue

            item_name = row['Item']
            item_id = item_mapping.get(item_name.lower())

            if item_id is None:
                unmapped_items.add(item_name)

            # Check if we have price data for this trade
            has_price_data = first_buy_time >= price_data_start

            trades.append({
                'first_buy_time': first_buy_time,
                'last_sell_time': parse_timestamp(row['Last sell time']),
                'account': row['Account'],
                'item_name': item_name,
                'item_id': item_id,
                'status': row['Status'],
                'bought': parse_int(row['Bought']) or 0,
                'sold': parse_int(row['Sold']) or 0,
                'avg_buy_price': parse_decimal(row['Avg. buy price']) or 0,
                'avg_sell_price': parse_decimal(row['Avg. sell price']),
                'tax': parse_decimal(row['Tax']),
                'profit': parse_decimal(row['Profit']),
                'profit_each': parse_decimal(row['Profit ea.']),
                'has_price_data': has_price_data,
            })

    print(f"Parsed {len(trades)} trades")
    print(f"Unmapped items ({len(unmapped_items)}): {sorted(unmapped_items)[:10]}...")

    # Clear existing data
    cur = conn.cursor()
    cur.execute("TRUNCATE training_trades RESTART IDENTITY CASCADE")
    conn.commit()

    # Insert trades
    insert_sql = """
        INSERT INTO training_trades (
            first_buy_time, last_sell_time, account, item_name, item_id,
            status, bought, sold, avg_buy_price, avg_sell_price,
            tax, profit, profit_each, has_price_data
        ) VALUES %s
    """

    values = [
        (
            t['first_buy_time'], t['last_sell_time'], t['account'], t['item_name'], t['item_id'],
            t['status'], t['bought'], t['sold'], t['avg_buy_price'], t['avg_sell_price'],
            t['tax'], t['profit'], t['profit_each'], t['has_price_data']
        )
        for t in trades
    ]

    execute_values(cur, insert_sql, values)
    conn.commit()

    # Report stats
    cur.execute("SELECT COUNT(*) FROM training_trades")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM training_trades WHERE status = 'FINISHED'")
    finished = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM training_trades WHERE has_price_data = TRUE AND status = 'FINISHED'")
    with_price_data = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM training_trades WHERE item_id IS NOT NULL AND status = 'FINISHED'")
    with_item_id = cur.fetchone()[0]

    cur.close()

    print(f"\nImport complete:")
    print(f"  Total trades: {total}")
    print(f"  Finished trades: {finished}")
    print(f"  With price data: {with_price_data}")
    print(f"  With item ID mapped: {with_item_id}")
    print(f"  Usable for training (finished + price data + item ID): TBD")

    return total


def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'osrs_data'),
        user=os.getenv('DB_USER', 'osrs_user'),
        password=os.getenv('DB_PASS', 'osrs_price_data_2024'),
    )

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'flips.csv')
    if not os.path.exists(csv_path):
        # Try alternative path on Hydra
        csv_path = os.path.expanduser('~/gept/data/flips.csv')

    print(f"Importing from: {csv_path}")
    import_csv(csv_path, conn)
    conn.close()


if __name__ == '__main__':
    main()
