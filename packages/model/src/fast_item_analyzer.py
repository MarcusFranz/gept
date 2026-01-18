"""
Fast Item Quality Analyzer

Uses the prices_1h table (1.5M rows) for quick item tiering,
then validates selected items against price_data_5min.
"""

import os
import psycopg2
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

from db_utils import CONN_PARAMS


def get_item_stats_fast() -> pd.DataFrame:
    """
    Get item statistics from prices_1h table (fast query).
    Returns DataFrame with per-item statistics.
    """
    conn = psycopg2.connect(**CONN_PARAMS)

    query = """
    WITH item_stats AS (
        SELECT
            item_id,
            COUNT(*) as row_count,
            MIN(timestamp) as min_date,
            MAX(timestamp) as max_date,
            SUM(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as total_volume,
            AVG(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as avg_volume,
            COUNT(CASE WHEN avg_high_price IS NOT NULL THEN 1 END) as high_count,
            COUNT(CASE WHEN avg_low_price IS NOT NULL THEN 1 END) as low_count,
            AVG(avg_high_price) as avg_high,
            AVG(avg_low_price) as avg_low
        FROM prices_1h
        GROUP BY item_id
    )
    SELECT
        s.*,
        i.name
    FROM item_stats s
    LEFT JOIN items i ON s.item_id = i.item_id
    ORDER BY s.total_volume DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def compute_item_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add tier assignments to item DataFrame."""
    df = df.copy()

    # Calculate derived metrics
    df['date_range_days'] = (pd.to_datetime(df['max_date']) -
                             pd.to_datetime(df['min_date'])).dt.total_seconds() / 86400
    df['months'] = df['date_range_days'] / 30

    # Expected hourly intervals
    df['expected_intervals'] = df['date_range_days'] * 24
    df['completeness'] = df['row_count'] / df['expected_intervals'].clip(lower=1)
    df['completeness'] = df['completeness'].clip(upper=1.0)

    # Price completeness
    df['high_completeness'] = df['high_count'] / df['row_count'].clip(lower=1)
    df['low_completeness'] = df['low_count'] / df['row_count'].clip(lower=1)
    df['price_completeness'] = (df['high_completeness'] + df['low_completeness']) / 2

    # Average price
    df['avg_price'] = (df['avg_high'].fillna(0) + df['avg_low'].fillna(0)) / 2

    # Spread
    df['spread_pct'] = ((df['avg_high'] - df['avg_low']) /
                        ((df['avg_high'] + df['avg_low']) / 2) * 100)

    # Tier assignment
    def assign_tier(row):
        # Tier 1: High quality
        if (row['completeness'] >= 0.95
                and row['avg_volume'] >= 1000
                and row['months'] >= 3
                and row['price_completeness'] >= 0.9):
            return 1

        # Tier 2: Medium quality
        if (row['completeness'] >= 0.80
                and row['avg_volume'] >= 100
                and row['months'] >= 2
                and row['price_completeness'] >= 0.7):
            return 2

        # Tier 3: Low quality
        if (row['completeness'] >= 0.50
                and row['avg_volume'] >= 10
                and row['months'] >= 1):
            return 3

        # Tier 4: Insufficient
        return 4

    df['tier'] = df.apply(assign_tier, axis=1)

    return df


def validate_5min_sample(item_ids: List[int], sample_size: int = 10000) -> Dict[int, Dict]:
    """
    Validate selected items against price_data_5min table.
    Returns dict with validation results per item.
    """
    conn = psycopg2.connect(**CONN_PARAMS)
    cursor = conn.cursor()

    results = {}
    for item_id in item_ids:
        try:
            # Get sample from 5min data
            cursor.execute("""
                SELECT
                    COUNT(*) as total_rows,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    SUM(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as total_volume,
                    AVG(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as avg_volume,
                    COUNT(CASE WHEN avg_high_price IS NOT NULL AND avg_low_price IS NOT NULL THEN 1 END) as complete_rows
                FROM price_data_5min
                WHERE item_id = %s
            """, (item_id,))

            row = cursor.fetchone()
            if row and row[0] > 0:
                total_rows = row[0]
                min_date = row[1]
                max_date = row[2]
                date_range = (max_date - min_date).total_seconds() / 86400 if min_date and max_date else 0
                expected_intervals = date_range * 288  # 5min intervals per day

                results[item_id] = {
                    'total_rows': total_rows,
                    'min_date': str(min_date),
                    'max_date': str(max_date),
                    'date_range_days': round(date_range, 1),
                    'expected_intervals': int(expected_intervals),
                    'completeness': round(total_rows / max(1, expected_intervals), 4),
                    'total_volume': int(row[3] or 0),
                    'avg_volume': round(row[4] or 0, 2),
                    'complete_rows': row[5],
                    'price_completeness': round(row[5] / max(1, total_rows), 4)
                }
            else:
                results[item_id] = {'error': 'No data found'}

        except Exception as e:
            results[item_id] = {'error': str(e)}

    cursor.close()
    conn.close()
    return results


def main():
    """Run fast item analysis."""
    print("="*60)
    print("FAST ITEM QUALITY ANALYSIS")
    print("="*60)

    os.makedirs('data', exist_ok=True)

    # Step 1: Get stats from 1h table (fast)
    print("\n1. Getting item statistics from prices_1h...")
    df = get_item_stats_fast()
    print(f"   Found {len(df)} items with hourly data")

    # Step 2: Compute tiers
    print("\n2. Computing tier assignments...")
    df = compute_item_tiers(df)

    tier_counts = df['tier'].value_counts().sort_index()
    print(f"   Tier 1 (High):     {tier_counts.get(1, 0):5d} items")
    print(f"   Tier 2 (Medium):   {tier_counts.get(2, 0):5d} items")
    print(f"   Tier 3 (Low):      {tier_counts.get(3, 0):5d} items")
    print(f"   Tier 4 (Skip):     {tier_counts.get(4, 0):5d} items")

    # Step 3: Validate top tier 1 items against 5min data
    tier_1_items = df[df['tier'] == 1]['item_id'].tolist()
    print(f"\n3. Validating {len(tier_1_items)} Tier 1 items against 5min data...")

    if tier_1_items:
        validation = validate_5min_sample(tier_1_items[:50])  # Validate top 50

        # Update tier based on 5min validation
        for item_id, val in validation.items():
            if 'error' not in val:
                if val['completeness'] < 0.90 or val['price_completeness'] < 0.7:
                    # Downgrade to tier 2
                    df.loc[df['item_id'] == item_id, 'tier'] = 2

    # Step 4: Save results
    print("\n4. Saving results...")

    # Convert timestamp columns to strings for JSON
    for col in ['min_date', 'max_date']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Full analysis
    df.to_json('data/item_analysis.json', orient='records', indent=2)
    print("   Full analysis: data/item_analysis.json")

    # Tier-specific lists
    for tier in [1, 2, 3]:
        tier_df = df[df['tier'] == tier].sort_values('avg_volume', ascending=False)
        tier_df.to_json(f'data/tier_{tier}_items.json', orient='records', indent=2)
        print(f"   Tier {tier}: data/tier_{tier}_items.json ({len(tier_df)} items)")

    # Summary for training
    trainable = df[df['tier'].isin([1, 2])]
    summary = {
        'total_items': len(df),
        'tier_1_count': int(tier_counts.get(1, 0)),
        'tier_2_count': int(tier_counts.get(2, 0)),
        'tier_3_count': int(tier_counts.get(3, 0)),
        'tier_4_count': int(tier_counts.get(4, 0)),
        'trainable_items': len(trainable),
        'analysis_timestamp': datetime.now().isoformat()
    }
    with open('data/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("   Summary: data/analysis_summary.json")

    # Print top items
    print("\n" + "="*60)
    print("TOP 25 TIER 1 ITEMS (by average volume)")
    print("="*60)
    print(f"{'ID':>8} {'Name':<30} {'AvgVol':>10} {'Cmpl%':>8} {'AvgPrice':>12}")
    print("-"*72)

    top_tier1 = df[df['tier'] == 1].nlargest(25, 'avg_volume')
    for _, row in top_tier1.iterrows():
        name = str(row['name'])[:30] if row['name'] else f"ID-{row['item_id']}"
        print(f"{row['item_id']:>8} {name:<30} {row['avg_volume']:>10,.0f} "
              f"{row['completeness']*100:>7.1f}% {row['avg_price']:>12,.0f}")

    print("\n" + "="*60)
    print("TOP 25 TIER 2 ITEMS (by average volume)")
    print("="*60)
    top_tier2 = df[df['tier'] == 2].nlargest(25, 'avg_volume')
    for _, row in top_tier2.iterrows():
        name = str(row['name'])[:30] if row['name'] else f"ID-{row['item_id']}"
        print(f"{row['item_id']:>8} {name:<30} {row['avg_volume']:>10,.0f} "
              f"{row['completeness']*100:>7.1f}% {row['avg_price']:>12,.0f}")

    return df


if __name__ == "__main__":
    main()
