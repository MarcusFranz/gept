#!/usr/bin/env python3
"""
ML Ranker Validation Script

Analyzes shadow ranking logs to compare ML ranker vs heuristic ranker performance.
For each logged recommendation, checks if the predicted fill would have occurred
based on actual price movements in the database.

Usage:
    python validate_ml_ranker.py                    # Validate today's data
    python validate_ml_ranker.py --date 2026-01-19  # Validate specific date
    python validate_ml_ranker.py --days 7           # Validate last 7 days
    python validate_ml_ranker.py --output report.json  # Save detailed report
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml-validator")

# Configuration
SHADOW_LOG_DIR = Path(os.path.expanduser("~/gept/logs/shadow_ranking"))
VALIDATION_WINDOWS = [1, 2, 4, 8, 12, 24]  # Hours to check for fill
TOP_N = 5  # Compare top N picks from each ranker


@dataclass
class ValidationResult:
    """Result of validating a single recommendation."""
    item_id: int
    item_name: str
    timestamp: datetime
    buy_price: float
    sell_price: float
    heuristic_rank: int
    ml_rank: int
    in_heuristic_top5: bool
    in_ml_top5: bool
    # Validation results per window
    filled_1h: bool = False
    filled_2h: bool = False
    filled_4h: bool = False
    filled_8h: bool = False
    filled_12h: bool = False
    filled_24h: bool = False
    actual_high_1h: Optional[float] = None
    actual_high_24h: Optional[float] = None


@dataclass
class ValidationSummary:
    """Summary statistics for validation."""
    total_recommendations: int
    unique_items: int
    date_range: str

    # Heuristic top N stats
    heuristic_top_n_count: int
    heuristic_fill_rate_1h: float
    heuristic_fill_rate_4h: float
    heuristic_fill_rate_24h: float

    # ML top N stats
    ml_top_n_count: int
    ml_fill_rate_1h: float
    ml_fill_rate_4h: float
    ml_fill_rate_24h: float

    # Comparison
    ml_improvement_1h: float  # Percentage points
    ml_improvement_4h: float
    ml_improvement_24h: float

    # Agreement stats
    both_top_n_count: int
    only_heuristic_count: int
    only_ml_count: int


def get_db_connection():
    """Get database connection using environment variables."""
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
        database=os.environ.get("DB_NAME", "osrs_data"),
        user=os.environ.get("DB_USER", "osrs_user"),
        password=os.environ.get("DB_PASS", ""),
    )


def load_shadow_log(date_str: str) -> list[dict]:
    """Load shadow ranking log for a specific date."""
    log_file = SHADOW_LOG_DIR / f"shadow_ranking_{date_str}.csv"

    if not log_file.exists():
        logger.warning(f"Shadow log not found: {log_file}")
        return []

    entries = []
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    logger.info(f"Loaded {len(entries)} entries from {log_file}")
    return entries


def check_fill(
    conn,
    item_id: int,
    timestamp: datetime,
    sell_price: float,
    window_hours: int
) -> tuple[bool, Optional[float]]:
    """
    Check if the sell price was reached within the time window.

    Returns (filled, actual_high_price)
    """
    end_time = timestamp + timedelta(hours=window_hours)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(avg_high_price) as max_high
            FROM price_data_5min
            WHERE item_id = %s
            AND timestamp > %s
            AND timestamp <= %s
            AND avg_high_price IS NOT NULL
        """, (item_id, timestamp, end_time))

        result = cur.fetchone()
        if result and result[0]:
            actual_high = float(result[0])
            filled = actual_high >= sell_price
            return filled, actual_high

        return False, None


def validate_entry(conn, entry: dict) -> Optional[ValidationResult]:
    """Validate a single shadow log entry."""
    try:
        timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        item_id = int(entry["item_id"])
        buy_price = float(entry["buy_price"])
        sell_price = float(entry["sell_price"])
        heuristic_rank = int(entry["heuristic_rank"])
        ml_rank = int(entry["ml_rank"])
        in_heuristic_top5 = entry["in_heuristic_top5"] == "1"
        in_ml_top5 = entry["in_ml_top5"] == "1"

        result = ValidationResult(
            item_id=item_id,
            item_name=entry["item_name"],
            timestamp=timestamp,
            buy_price=buy_price,
            sell_price=sell_price,
            heuristic_rank=heuristic_rank,
            ml_rank=ml_rank,
            in_heuristic_top5=in_heuristic_top5,
            in_ml_top5=in_ml_top5,
        )

        # Check fills for each window
        for window in VALIDATION_WINDOWS:
            filled, actual_high = check_fill(conn, item_id, timestamp, sell_price, window)
            setattr(result, f"filled_{window}h", filled)
            if window == 1:
                result.actual_high_1h = actual_high
            elif window == 24:
                result.actual_high_24h = actual_high

        return result

    except (KeyError, ValueError) as e:
        logger.debug(f"Skipping invalid entry: {e}")
        return None


def calculate_summary(results: list[ValidationResult], date_range: str) -> ValidationSummary:
    """Calculate summary statistics from validation results."""

    # Filter to unique item recommendations (dedupe by item_id + hour)
    seen = set()
    unique_results = []
    for r in results:
        key = (r.item_id, r.timestamp.strftime("%Y-%m-%d %H"))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    heuristic_top_n = [r for r in unique_results if r.in_heuristic_top5]
    ml_top_n = [r for r in unique_results if r.in_ml_top5]
    both_top_n = [r for r in unique_results if r.in_heuristic_top5 and r.in_ml_top5]
    only_heuristic = [r for r in unique_results if r.in_heuristic_top5 and not r.in_ml_top5]
    only_ml = [r for r in unique_results if r.in_ml_top5 and not r.in_heuristic_top5]

    def fill_rate(entries: list, window: str) -> float:
        if not entries:
            return 0.0
        attr = f"filled_{window}"
        filled = sum(1 for e in entries if getattr(e, attr))
        return filled / len(entries)

    h_1h = fill_rate(heuristic_top_n, "1h")
    h_4h = fill_rate(heuristic_top_n, "4h")
    h_24h = fill_rate(heuristic_top_n, "24h")

    m_1h = fill_rate(ml_top_n, "1h")
    m_4h = fill_rate(ml_top_n, "4h")
    m_24h = fill_rate(ml_top_n, "24h")

    return ValidationSummary(
        total_recommendations=len(unique_results),
        unique_items=len(set(r.item_id for r in unique_results)),
        date_range=date_range,
        heuristic_top_n_count=len(heuristic_top_n),
        heuristic_fill_rate_1h=h_1h,
        heuristic_fill_rate_4h=h_4h,
        heuristic_fill_rate_24h=h_24h,
        ml_top_n_count=len(ml_top_n),
        ml_fill_rate_1h=m_1h,
        ml_fill_rate_4h=m_4h,
        ml_fill_rate_24h=m_24h,
        ml_improvement_1h=(m_1h - h_1h) * 100,
        ml_improvement_4h=(m_4h - h_4h) * 100,
        ml_improvement_24h=(m_24h - h_24h) * 100,
        both_top_n_count=len(both_top_n),
        only_heuristic_count=len(only_heuristic),
        only_ml_count=len(only_ml),
    )


def print_summary(summary: ValidationSummary):
    """Print validation summary to console."""
    print("\n" + "=" * 60)
    print("ML RANKER VALIDATION REPORT")
    print("=" * 60)
    print(f"Date Range: {summary.date_range}")
    print(f"Total Recommendations: {summary.total_recommendations}")
    print(f"Unique Items: {summary.unique_items}")
    print()

    print("FILL RATES (Top 5 Picks)")
    print("-" * 60)
    print(f"{'Window':<12} {'Heuristic':<15} {'ML Ranker':<15} {'ML Δ':<10}")
    print("-" * 60)

    for window, h_rate, m_rate, improvement in [
        ("1 hour", summary.heuristic_fill_rate_1h, summary.ml_fill_rate_1h, summary.ml_improvement_1h),
        ("4 hours", summary.heuristic_fill_rate_4h, summary.ml_fill_rate_4h, summary.ml_improvement_4h),
        ("24 hours", summary.heuristic_fill_rate_24h, summary.ml_fill_rate_24h, summary.ml_improvement_24h),
    ]:
        h_pct = f"{h_rate * 100:.1f}%"
        m_pct = f"{m_rate * 100:.1f}%"
        delta = f"{improvement:+.1f}pp"
        indicator = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
        print(f"{window:<12} {h_pct:<15} {m_pct:<15} {delta:<10} {indicator}")

    print()
    print("RANKER AGREEMENT")
    print("-" * 60)
    print(f"Both in top 5:        {summary.both_top_n_count}")
    print(f"Only heuristic top 5: {summary.only_heuristic_count}")
    print(f"Only ML top 5:        {summary.only_ml_count}")
    print()

    # Verdict
    avg_improvement = (summary.ml_improvement_1h + summary.ml_improvement_4h + summary.ml_improvement_24h) / 3
    if avg_improvement > 2:
        verdict = "✓ ML ranker shows IMPROVEMENT over heuristic"
    elif avg_improvement < -2:
        verdict = "✗ ML ranker UNDERPERFORMS heuristic"
    else:
        verdict = "= ML ranker performs SIMILARLY to heuristic"

    print(f"VERDICT: {verdict}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate ML ranker against actual price movements")
    parser.add_argument("--date", help="Specific date to validate (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to validate (default: 1)")
    parser.add_argument("--output", help="Output file for detailed JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine dates to validate
    if args.date:
        dates = [args.date]
    else:
        today = datetime.now(timezone.utc)
        # Start from yesterday (today's data may not have enough price history)
        dates = [
            (today - timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(args.days)
        ]

    logger.info(f"Validating dates: {dates}")

    # Load all shadow log entries
    all_entries = []
    for date_str in dates:
        entries = load_shadow_log(date_str)
        all_entries.extend(entries)

    if not all_entries:
        logger.error("No shadow log entries found")
        sys.exit(1)

    logger.info(f"Loaded {len(all_entries)} total entries")

    # Connect to database and validate
    conn = get_db_connection()

    try:
        results = []
        total = len(all_entries)

        for i, entry in enumerate(all_entries):
            if i % 1000 == 0:
                logger.info(f"Validating {i}/{total}...")

            result = validate_entry(conn, entry)
            if result:
                results.append(result)

        logger.info(f"Validated {len(results)} entries")

        # Calculate and print summary
        date_range = f"{dates[-1]} to {dates[0]}" if len(dates) > 1 else dates[0]
        summary = calculate_summary(results, date_range)
        print_summary(summary)

        # Save detailed report if requested
        if args.output:
            report = {
                "summary": asdict(summary),
                "results": [
                    {
                        "item_id": r.item_id,
                        "item_name": r.item_name,
                        "timestamp": r.timestamp.isoformat(),
                        "heuristic_rank": r.heuristic_rank,
                        "ml_rank": r.ml_rank,
                        "in_heuristic_top5": r.in_heuristic_top5,
                        "in_ml_top5": r.in_ml_top5,
                        "filled_1h": r.filled_1h,
                        "filled_4h": r.filled_4h,
                        "filled_24h": r.filled_24h,
                    }
                    for r in results
                ]
            }

            with open(args.output, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Detailed report saved to {args.output}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
