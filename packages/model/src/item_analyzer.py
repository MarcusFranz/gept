"""
Item Quality Analyzer for GE Flipping

Analyzes items from price_data_5min table and creates tiers for training.
Uses efficient batched queries to handle the 426M row table.
"""

import os
import psycopg2
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional

from db_utils import CONN_PARAMS

# Tier thresholds based on actual data quality (5min data)
# Data shows ~54% completeness is typical for long-running items
TIER_THRESHOLDS = {
    1: {
        'min_completeness': 0.40,     # 40%+ of expected intervals
        'min_avg_volume': 50000,      # High volume items
        'min_days': 365,              # At least 1 year
        'min_price_completeness': 0.7  # 70% of rows have both prices
    },
    2: {
        'min_completeness': 0.30,
        'min_avg_volume': 10000,
        'min_days': 180,
        'min_price_completeness': 0.5
    },
    3: {
        'min_completeness': 0.15,
        'min_avg_volume': 1000,
        'min_days': 30,
        'min_price_completeness': 0.3
    }
}


def get_all_item_ids() -> List[int]:
    """Get list of all tradeable items."""
    conn = psycopg2.connect(**CONN_PARAMS)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT item_id, name FROM items
        WHERE tradeable = true
        ORDER BY item_id
    """)
    items = [(row[0], row[1]) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return items


def analyze_item(item_id: int, conn) -> Optional[Dict]:
    """Analyze a single item's data quality."""
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                COUNT(*) as row_count,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                SUM(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as total_volume,
                AVG(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as avg_volume,
                COUNT(CASE WHEN avg_high_price IS NOT NULL THEN 1 END) as high_count,
                COUNT(CASE WHEN avg_low_price IS NOT NULL THEN 1 END) as low_count,
                COUNT(CASE WHEN avg_high_price IS NOT NULL AND avg_low_price IS NOT NULL THEN 1 END) as both_count,
                AVG(avg_high_price) as avg_high,
                AVG(avg_low_price) as avg_low,
                STDDEV(avg_high_price) as std_high,
                STDDEV(avg_low_price) as std_low
            FROM price_data_5min
            WHERE item_id = %s
        """, (item_id,))

        row = cursor.fetchone()
        cursor.close()

        if not row or row[0] == 0:
            return None

        row_count = row[0]
        min_date = row[1]
        max_date = row[2]
        total_volume = row[3] or 0
        avg_volume = row[4] or 0
        # high_count and low_count not used separately, only both_count matters
        both_count = row[7]
        avg_high = row[8]
        avg_low = row[9]
        std_high = row[10]
        std_low = row[11]

        # Calculate metrics
        date_range_days = (max_date - min_date).total_seconds() / 86400 if min_date and max_date else 0
        expected_intervals = date_range_days * 288  # 5min intervals per day
        completeness = row_count / max(1, expected_intervals)

        price_completeness = both_count / max(1, row_count)

        avg_price = (avg_high + avg_low) / 2 if avg_high and avg_low else (avg_high or avg_low or 0)
        spread_pct = ((avg_high - avg_low) / avg_price * 100) if avg_price > 0 and avg_high and avg_low else 0

        # Coefficient of variation (volatility indicator)
        cv_high = (std_high / avg_high * 100) if avg_high and std_high else 0
        cv_low = (std_low / avg_low * 100) if avg_low and std_low else 0

        # Assign tier
        tier = 4  # Default: insufficient
        for t in [1, 2, 3]:
            thresh = TIER_THRESHOLDS[t]
            if (completeness >= thresh['min_completeness']
                    and avg_volume >= thresh['min_avg_volume']
                    and date_range_days >= thresh['min_days']
                    and price_completeness >= thresh['min_price_completeness']):
                tier = t
                break

        return {
            'item_id': item_id,
            'row_count': row_count,
            'min_date': str(min_date) if min_date else None,
            'max_date': str(max_date) if max_date else None,
            'date_range_days': round(date_range_days, 1),
            'months': round(date_range_days / 30, 1),
            'completeness': round(completeness, 4),
            'price_completeness': round(price_completeness, 4),
            'total_volume': int(total_volume),
            'avg_volume': round(avg_volume, 2),
            'avg_high_price': round(avg_high, 2) if avg_high else None,
            'avg_low_price': round(avg_low, 2) if avg_low else None,
            'avg_price': round(avg_price, 2),
            'spread_pct': round(spread_pct, 4),
            'volatility_high': round(cv_high, 2),
            'volatility_low': round(cv_low, 2),
            'tier': tier
        }

    except Exception as e:
        cursor.close()
        return {'item_id': item_id, 'error': str(e), 'tier': 4}


def analyze_top_items(limit: int = 500) -> List[Dict]:
    """
    Analyze top items by volume (more efficient approach).

    First identifies high-volume items from a sample, then analyzes them in detail.
    """
    print(f"Analyzing top {limit} items by volume...")

    conn = psycopg2.connect(**CONN_PARAMS)
    cursor = conn.cursor()

    # Get top items by total volume (using index on item_id)
    # This query runs relatively fast because it's aggregated
    print("  Step 1: Identifying high-volume items...")
    cursor.execute("""
        WITH item_volumes AS (
            SELECT
                item_id,
                SUM(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as total_vol
            FROM price_data_5min
            GROUP BY item_id
            HAVING SUM(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) > 0
            ORDER BY total_vol DESC
            LIMIT %s
        )
        SELECT iv.item_id, iv.total_vol, i.name
        FROM item_volumes iv
        LEFT JOIN items i ON iv.item_id = i.item_id
        ORDER BY iv.total_vol DESC
    """, (limit,))

    top_items = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    print(f"  Found {len(top_items)} items with trading volume")
    cursor.close()

    # Now analyze each item in detail
    print("  Step 2: Analyzing item data quality...")
    results = []
    for i, (item_id, total_vol, name) in enumerate(top_items):
        result = analyze_item(item_id, conn)
        if result:
            result['name'] = name or f"Unknown-{item_id}"
            results.append(result)

        if (i + 1) % 50 == 0:
            print(f"    Analyzed {i+1}/{len(top_items)} items...")

    conn.close()
    return results


def save_results(results: List[Dict], output_dir: str = 'data'):
    """Save analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Save full results
    df.to_json(f'{output_dir}/item_analysis_full.json', orient='records', indent=2)

    # Save tier-specific files
    tier_counts = {}
    for tier in [1, 2, 3, 4]:
        tier_df = df[df['tier'] == tier].sort_values('avg_volume', ascending=False)
        tier_df.to_json(f'{output_dir}/tier_{tier}_items.json', orient='records', indent=2)
        tier_counts[tier] = len(tier_df)

    # Save summary
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_items_analyzed': len(df),
        'tier_counts': tier_counts,
        'tier_thresholds': TIER_THRESHOLDS,
        'trainable_items': tier_counts.get(1, 0) + tier_counts.get(2, 0)
    }
    with open(f'{output_dir}/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return df, summary


def main():
    """Main analysis entry point."""
    print("="*70)
    print("OSRS ITEM DATA QUALITY ANALYSIS")
    print("="*70)

    # Analyze top 500 items by volume
    results = analyze_top_items(limit=500)

    # Save results
    df, summary = save_results(results)

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total items analyzed: {summary['total_items_analyzed']}")
    print("\nTier distribution:")
    print(f"  Tier 1 (High quality):   {summary['tier_counts'].get(1, 0):4d} items")
    print(f"  Tier 2 (Medium quality): {summary['tier_counts'].get(2, 0):4d} items")
    print(f"  Tier 3 (Low quality):    {summary['tier_counts'].get(3, 0):4d} items")
    print(f"  Tier 4 (Insufficient):   {summary['tier_counts'].get(4, 0):4d} items")
    print(f"\nTrainable items (Tier 1+2): {summary['trainable_items']}")

    # Print top tier 1 items
    tier_1 = df[df['tier'] == 1].nlargest(30, 'avg_volume')
    if len(tier_1) > 0:
        print("\n" + "="*70)
        print("TOP 30 TIER 1 ITEMS")
        print("="*70)
        print(f"{'ID':>8} {'Name':<25} {'AvgVol':>12} {'Cmpl%':>8} {'Days':>6} {'AvgPrice':>12}")
        print("-"*75)
        for _, row in tier_1.iterrows():
            name = str(row['name'])[:25]
            print(f"{row['item_id']:>8} {name:<25} {row['avg_volume']:>12,.0f} "
                  f"{row['completeness']*100:>7.1f}% {row['date_range_days']:>6.0f} "
                  f"{row['avg_price']:>12,.0f}")

    # Print top tier 2 items
    tier_2 = df[df['tier'] == 2].nlargest(20, 'avg_volume')
    if len(tier_2) > 0:
        print("\n" + "="*70)
        print("TOP 20 TIER 2 ITEMS")
        print("="*70)
        for _, row in tier_2.iterrows():
            name = str(row['name'])[:25]
            print(f"{row['item_id']:>8} {name:<25} {row['avg_volume']:>12,.0f} "
                  f"{row['completeness']*100:>7.1f}% {row['date_range_days']:>6.0f}")

    print("\n" + "="*70)
    print("Files saved to data/ directory")
    print("="*70)

    return df


if __name__ == "__main__":
    main()
