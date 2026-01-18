#!/usr/bin/env python3
"""
Migrate player_counts data from DuckDB to PostgreSQL.

Usage:
    python migrate_player_counts.py
    python migrate_player_counts.py --dry-run  # Preview without inserting
"""

import os
import sys
import argparse

import duckdb
import psycopg2
from psycopg2.extras import execute_values

# DuckDB source
DUCKDB_PATH = "/home/ubuntu/osrs_collector/data/player_counts.duckdb"

# PostgreSQL target
PG_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'dbname': os.getenv('DB_NAME', 'osrs_data'),
    'user': os.getenv('DB_USER', 'osrs_user'),
    'password': os.environ['DB_PASS']
}

BATCH_SIZE = 10000


def migrate(dry_run: bool = False):
    """Migrate all player count records from DuckDB to PostgreSQL."""

    print(f"Connecting to DuckDB: {DUCKDB_PATH}")
    duck_conn = duckdb.connect(DUCKDB_PATH, read_only=True)

    # Get total count
    total = duck_conn.execute("SELECT COUNT(*) FROM player_counts").fetchone()[0]
    print(f"Found {total:,} records to migrate")

    if dry_run:
        print("\n[DRY RUN] Would migrate the following sample:")
        sample = duck_conn.execute(
            "SELECT timestamp, count, fetched_at FROM player_counts ORDER BY timestamp LIMIT 5"
        ).fetchall()
        for row in sample:
            print(f"  {row}")
        duck_conn.close()
        return

    print(f"\nConnecting to PostgreSQL...")
    pg_conn = psycopg2.connect(**PG_CONFIG)

    # Migrate in batches
    offset = 0
    migrated = 0

    try:
        with pg_conn.cursor() as cur:
            while offset < total:
                # Fetch batch from DuckDB
                rows = duck_conn.execute(f"""
                    SELECT timestamp, count, fetched_at
                    FROM player_counts
                    ORDER BY timestamp
                    LIMIT {BATCH_SIZE} OFFSET {offset}
                """).fetchall()

                if not rows:
                    break

                # Insert into PostgreSQL with ON CONFLICT DO NOTHING
                execute_values(
                    cur,
                    """
                    INSERT INTO player_counts (timestamp, count, fetched_at)
                    VALUES %s
                    ON CONFLICT (timestamp) DO NOTHING
                    """,
                    rows,
                    template="(%s, %s, %s)"
                )
                pg_conn.commit()

                migrated += len(rows)
                offset += BATCH_SIZE
                print(f"  Migrated {migrated:,}/{total:,} records ({100*migrated/total:.1f}%)")

        print(f"\nMigration complete! {migrated:,} records migrated.")

        # Verify
        cur = pg_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM player_counts")
        pg_count = cur.fetchone()[0]
        print(f"PostgreSQL now has {pg_count:,} player count records.")

    finally:
        duck_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate player counts to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run)
