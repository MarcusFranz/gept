"""
Data Quality Analyzer for OSRS Price Data
Analyzes each item's data quality and assigns tiers for model training.
"""

import os
import psycopg2
import json
from datetime import datetime
from typing import Dict, List

from db_utils import get_conn_params

# Expected data parameters
# 5-min intervals: 288 intervals per day, ~105,120 per year
INTERVALS_PER_DAY = 288
MIN_DATE = datetime(2021, 3, 8)
MAX_DATE = datetime(2026, 1, 7)
TOTAL_EXPECTED_DAYS = (MAX_DATE - MIN_DATE).days
TOTAL_EXPECTED_INTERVALS = TOTAL_EXPECTED_DAYS * INTERVALS_PER_DAY

# Tier thresholds
TIER_CONFIG = {
    1: {'min_completeness': 0.95, 'min_volume_avg': 1000, 'min_months': 3},
    2: {'min_completeness': 0.80, 'min_volume_avg': 100, 'min_months': 2},
    3: {'min_completeness': 0.50, 'min_volume_avg': 10, 'min_months': 1},
}

def get_item_ids() -> List[int]:
    """Get list of all item IDs with price data."""
    conn = psycopg2.connect(**get_conn_params())
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT item_id FROM price_data_5min ORDER BY item_id")
    items = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return items

def get_item_names() -> Dict[int, str]:
    """Get mapping of item IDs to names."""
    conn = psycopg2.connect(**get_conn_params())
    cursor = conn.cursor()
    cursor.execute("SELECT item_id, name FROM items")
    names = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return names

def analyze_item_batch(item_ids: List[int], conn) -> List[Dict]:
    """Analyze data quality for a batch of items."""
    results = []

    for item_id in item_ids:
        try:
            # Get summary statistics using aggregated query
            query = """
                SELECT
                    COUNT(*) as row_count,
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date,
                    SUM(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as total_volume,
                    AVG(COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0)) as avg_volume,
                    COUNT(CASE WHEN avg_high_price IS NOT NULL THEN 1 END) as high_price_count,
                    COUNT(CASE WHEN avg_low_price IS NOT NULL THEN 1 END) as low_price_count,
                    AVG(avg_high_price) as avg_high,
                    AVG(avg_low_price) as avg_low
                FROM price_data_5min
                WHERE item_id = %s
            """
            cursor = conn.cursor()
            cursor.execute(query, (item_id,))
            row = cursor.fetchone()

            if row[0] == 0:  # No data
                results.append({
                    'item_id': item_id,
                    'row_count': 0,
                    'tier': 4
                })
                continue

            row_count = row[0]
            min_date = row[1]
            max_date = row[2]
            total_volume = row[3] or 0
            avg_volume = row[4] or 0
            high_price_count = row[5]
            low_price_count = row[6]
            avg_high = row[7]
            avg_low = row[8]

            # Calculate date range
            date_range_days = (max_date - min_date).total_seconds() / 86400 if min_date and max_date else 0
            expected_intervals = date_range_days * INTERVALS_PER_DAY if date_range_days > 0 else 1

            # Completeness metrics
            completeness = row_count / expected_intervals if expected_intervals > 0 else 0
            high_price_completeness = high_price_count / row_count if row_count > 0 else 0
            low_price_completeness = low_price_count / row_count if row_count > 0 else 0
            price_completeness = (high_price_count + low_price_count) / (2 * row_count) if row_count > 0 else 0

            # Average price for filtering
            avg_price = (avg_high + avg_low) / 2 if avg_high and avg_low else (avg_high or avg_low or 0)

            # Calculate spread
            spread_pct = ((avg_high - avg_low) / ((avg_high + avg_low) / 2) * 100) if avg_high and avg_low and avg_high + avg_low > 0 else 0

            # Determine tier
            months = date_range_days / 30 if date_range_days else 0

            tier = 4  # Default: insufficient
            if completeness >= 0.95 and avg_volume >= 1000 and months >= 3:
                tier = 1
            elif completeness >= 0.80 and avg_volume >= 100 and months >= 2:
                tier = 2
            elif completeness >= 0.50 and avg_volume >= 10 and months >= 1:
                tier = 3

            results.append({
                'item_id': item_id,
                'row_count': row_count,
                'min_date': str(min_date) if min_date else None,
                'max_date': str(max_date) if max_date else None,
                'date_range_days': round(date_range_days, 1),
                'months': round(months, 1),
                'completeness': round(completeness, 4),
                'high_price_completeness': round(high_price_completeness, 4),
                'low_price_completeness': round(low_price_completeness, 4),
                'price_completeness': round(price_completeness, 4),
                'total_volume': int(total_volume),
                'avg_volume': round(avg_volume, 2),
                'avg_high_price': round(avg_high, 2) if avg_high else None,
                'avg_low_price': round(avg_low, 2) if avg_low else None,
                'avg_price': round(avg_price, 2) if avg_price else None,
                'spread_pct': round(spread_pct, 2),
                'tier': tier
            })

            cursor.close()

        except Exception as e:
            results.append({
                'item_id': item_id,
                'error': str(e),
                'tier': 4
            })

    return results

def run_full_analysis(batch_size: int = 50) -> Dict:
    """Run full data quality analysis on all items."""
    print("Starting data quality analysis...")

    item_ids = get_item_ids()
    item_names = get_item_names()

    print(f"Analyzing {len(item_ids)} items in batches of {batch_size}...")

    all_results = []
    conn = psycopg2.connect(**get_conn_params())

    for i in range(0, len(item_ids), batch_size):
        batch = item_ids[i:i+batch_size]
        batch_results = analyze_item_batch(batch, conn)
        all_results.extend(batch_results)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(item_ids))}/{len(item_ids)} items...")

    conn.close()

    # Add item names
    for result in all_results:
        result['name'] = item_names.get(result['item_id'], f"Unknown-{result['item_id']}")

    # Calculate tier statistics
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for result in all_results:
        tier_counts[result['tier']] += 1

    summary = {
        'total_items': len(all_results),
        'tier_counts': tier_counts,
        'tier_1_items': [r for r in all_results if r['tier'] == 1],
        'tier_2_items': [r for r in all_results if r['tier'] == 2],
        'tier_3_items': [r for r in all_results if r['tier'] == 3],
        'tier_4_items': [r for r in all_results if r['tier'] == 4],
        'all_items': all_results,
        'analysis_timestamp': datetime.now().isoformat()
    }

    return summary

def main():
    """Main entry point."""
    # Create output directory
    os.makedirs('data', exist_ok=True)

    # Run analysis
    results = run_full_analysis(batch_size=100)

    # Print summary
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total items analyzed: {results['total_items']}")
    print("\nTier distribution:")
    print(f"  Tier 1 (High quality):   {results['tier_counts'][1]:4d} items")
    print(f"  Tier 2 (Medium quality): {results['tier_counts'][2]:4d} items")
    print(f"  Tier 3 (Low quality):    {results['tier_counts'][3]:4d} items")
    print(f"  Tier 4 (Insufficient):   {results['tier_counts'][4]:4d} items")

    # Save full results
    with open('data/item_quality_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to data/item_quality_analysis.json")

    # Save tier lists for easy access
    for tier in [1, 2, 3]:
        tier_items = sorted(results[f'tier_{tier}_items'],
                            key=lambda x: x.get('avg_volume', 0) or 0,
                            reverse=True)
        with open(f'data/tier_{tier}_items.json', 'w') as f:
            json.dump(tier_items, f, indent=2)
        print(f"Tier {tier} items saved to data/tier_{tier}_items.json")

    # Print top tier 1 items
    print("\n" + "="*60)
    print("TOP 20 TIER 1 ITEMS (by avg volume)")
    print("="*60)
    tier_1_sorted = sorted(results['tier_1_items'],
                           key=lambda x: x.get('avg_volume', 0) or 0,
                           reverse=True)[:20]
    print(f"{'ID':>8} {'Name':<30} {'AvgVol':>12} {'Complete':>10} {'AvgPrice':>12}")
    print("-"*76)
    for item in tier_1_sorted:
        print(f"{item['item_id']:>8} {item['name'][:30]:<30} {item.get('avg_volume', 0):>12,.0f} "
              f"{item.get('completeness', 0)*100:>9.1f}% {item.get('avg_price', 0):>12,.0f}")

    return results

if __name__ == "__main__":
    main()
