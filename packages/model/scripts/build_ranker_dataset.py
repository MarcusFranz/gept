#!/usr/bin/env python3
"""Build training dataset for the ML ranker.

This script:
1. Loads trades with price data coverage
2. Computes market features at trade time
3. Joins with item properties
4. Exports as a training-ready dataset
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import psycopg2
from tqdm import tqdm

# Add src to path for feature engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from feature_engine import FeatureEngine, Granularity
except ImportError:
    print("Warning: Could not import FeatureEngine, will compute basic features only")
    FeatureEngine = None


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'osrs_data'),
        user=os.getenv('DB_USER', 'osrs_user'),
        password=os.getenv('DB_PASS', 'osrs_price_data_2024'),
    )


def load_usable_trades(conn) -> pd.DataFrame:
    """Load finished trades with price data coverage and valid item_id."""
    query = """
        SELECT
            t.id,
            t.first_buy_time,
            t.last_sell_time,
            t.item_id,
            t.item_name,
            t.bought,
            t.sold,
            t.avg_buy_price,
            t.avg_sell_price,
            t.tax,
            t.profit,
            t.profit_each
        FROM training_trades t
        WHERE t.status = 'FINISHED'
          AND t.has_price_data = TRUE
          AND t.item_id IS NOT NULL
          AND t.profit IS NOT NULL
        ORDER BY t.first_buy_time
    """
    return pd.read_sql(query, conn)


def load_item_properties(conn) -> pd.DataFrame:
    """Load item properties for all tradeable items."""
    query = """
        SELECT
            item_id,
            name,
            buy_limit,
            members,
            highalch,
            lowalch
        FROM items
        WHERE tradeable = TRUE
    """
    return pd.read_sql(query, conn)


def compute_market_features_for_trade(conn, item_id: int, trade_time: datetime) -> dict:
    """Compute market features at the time of trade.

    Features computed:
    - price_trend_1h: Price change over last hour
    - price_trend_24h: Price change over last 24 hours
    - volatility_24h: Price volatility over last 24 hours
    - volume_24h: Total volume over last 24 hours
    - spread_pct: Current bid-ask spread as percentage
    - avg_price: Average mid price
    """
    cur = conn.cursor()

    # Get price data for the 24 hours before the trade
    lookback_start = trade_time - timedelta(hours=24)

    cur.execute("""
        SELECT
            timestamp,
            avg_high_price,
            avg_low_price,
            high_price_volume,
            low_price_volume
        FROM price_data_5min
        WHERE item_id = %s
          AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp
    """, (item_id, lookback_start, trade_time))

    rows = cur.fetchall()
    cur.close()

    if len(rows) < 12:  # Need at least 1 hour of data
        return None

    df = pd.DataFrame(rows, columns=['timestamp', 'high', 'low', 'high_vol', 'low_vol'])

    # Handle NaN prices
    df['high'] = df['high'].ffill().bfill()
    df['low'] = df['low'].ffill().bfill()
    df['mid'] = (df['high'] + df['low']) / 2

    # Current values (last row)
    current = df.iloc[-1]

    # 1 hour ago (12 5-min periods)
    hour_ago_idx = max(0, len(df) - 12)
    hour_ago = df.iloc[hour_ago_idx]

    # 24 hours ago (first row if we have full 24h)
    day_ago = df.iloc[0]

    features = {
        'avg_price': float(current['mid']),
        'spread_pct': float((current['high'] - current['low']) / current['mid']) if current['mid'] > 0 else 0,
        'price_trend_1h': float((current['mid'] - hour_ago['mid']) / hour_ago['mid']) if hour_ago['mid'] > 0 else 0,
        'price_trend_24h': float((current['mid'] - day_ago['mid']) / day_ago['mid']) if day_ago['mid'] > 0 else 0,
        'volatility_24h': float(df['mid'].std() / df['mid'].mean()) if df['mid'].mean() > 0 else 0,
        'volume_24h': float(df['high_vol'].sum() + df['low_vol'].sum()),
    }

    return features


def compute_time_features(trade_time: datetime) -> dict:
    """Compute time-based features."""
    return {
        'hour_of_day': trade_time.hour,
        'day_of_week': trade_time.weekday(),
        'is_weekend': 1 if trade_time.weekday() >= 5 else 0,
    }


def compute_trade_features(trade: pd.Series) -> dict:
    """Compute features from the trade itself."""
    return {
        'trade_quantity': int(trade['bought']),
        'trade_margin_pct': float((trade['avg_sell_price'] - trade['avg_buy_price']) / trade['avg_buy_price'])
            if trade['avg_buy_price'] > 0 else 0,
        'trade_capital': float(trade['avg_buy_price'] * trade['bought']),
    }


def build_dataset(conn, trades_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """Build complete training dataset with all features."""
    print(f"Building dataset for {len(trades_df)} trades...")

    # Pre-compute item property lookup
    item_props = items_df.set_index('item_id').to_dict('index')

    records = []
    failed_count = 0

    for _, trade in tqdm(trades_df.iterrows(), total=len(trades_df), desc="Computing features"):
        item_id = trade['item_id']
        trade_time = trade['first_buy_time']

        # Get market features
        market_features = compute_market_features_for_trade(conn, item_id, trade_time)
        if market_features is None:
            failed_count += 1
            continue

        # Get item properties
        item_info = item_props.get(item_id, {})
        item_features = {
            'buy_limit': item_info.get('buy_limit', 0) or 0,
            'is_members': 1 if item_info.get('members', False) else 0,
            'highalch': item_info.get('highalch', 0) or 0,
        }

        # Get time features
        time_features = compute_time_features(trade_time)

        # Get trade features
        trade_features = compute_trade_features(trade)

        # Combine all features
        record = {
            'trade_id': trade['id'],
            'item_id': item_id,
            'item_name': trade['item_name'],
            'first_buy_time': trade_time,
            'profit': float(trade['profit']),
            'profit_pct': float(trade['profit'] / (trade['avg_buy_price'] * trade['bought']))
                if trade['avg_buy_price'] * trade['bought'] > 0 else 0,
            **item_features,
            **market_features,
            **time_features,
            **trade_features,
        }

        records.append(record)

    print(f"Built {len(records)} feature records ({failed_count} failed)")
    return pd.DataFrame(records)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived/normalized features."""
    df = df.copy()

    # Log-transform skewed features
    df['log_buy_limit'] = np.log1p(df['buy_limit'])
    df['log_volume_24h'] = np.log1p(df['volume_24h'])
    df['log_avg_price'] = np.log1p(df['avg_price'])
    df['log_trade_capital'] = np.log1p(df['trade_capital'])

    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def create_pairwise_dataset(df: pd.DataFrame, pairs_per_trade: int = 5) -> pd.DataFrame:
    """Create pairwise comparisons for ranking training.

    For each trade, sample other trades from similar time periods
    and create pairs (trade_a, trade_b, a_is_better).
    """
    print(f"Creating pairwise dataset with {pairs_per_trade} pairs per trade...")

    # Sort by time
    df = df.sort_values('first_buy_time').reset_index(drop=True)

    pairs = []

    for i in tqdm(range(len(df)), desc="Creating pairs"):
        trade_a = df.iloc[i]

        # Sample from trades within 7 days
        window_start = max(0, i - 100)
        window_end = min(len(df), i + 100)

        candidates = list(range(window_start, window_end))
        candidates.remove(i)

        if len(candidates) == 0:
            continue

        # Sample pairs
        n_samples = min(pairs_per_trade, len(candidates))
        sampled = np.random.choice(candidates, size=n_samples, replace=False)

        for j in sampled:
            trade_b = df.iloc[j]

            # Determine which is better (higher profit)
            a_is_better = 1 if trade_a['profit'] > trade_b['profit'] else 0

            pairs.append({
                'trade_a_id': trade_a['trade_id'],
                'trade_b_id': trade_b['trade_id'],
                'profit_a': trade_a['profit'],
                'profit_b': trade_b['profit'],
                'a_is_better': a_is_better,
            })

    print(f"Created {len(pairs)} pairs")
    return pd.DataFrame(pairs)


def main():
    conn = get_db_connection()

    # Load data
    print("Loading trades...")
    trades_df = load_usable_trades(conn)
    print(f"Loaded {len(trades_df)} usable trades")

    print("Loading item properties...")
    items_df = load_item_properties(conn)
    print(f"Loaded {len(items_df)} items")

    # Build feature dataset
    dataset = build_dataset(conn, trades_df, items_df)

    # Add derived features
    dataset = add_derived_features(dataset)

    # Save feature dataset
    output_dir = os.path.expanduser('~/gept/data/ranker_training')
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, 'trade_features.parquet')
    dataset.to_parquet(features_path, index=False)
    print(f"Saved features to {features_path}")

    # Create pairwise dataset
    pairs_df = create_pairwise_dataset(dataset)
    pairs_path = os.path.join(output_dir, 'trade_pairs.parquet')
    pairs_df.to_parquet(pairs_path, index=False)
    print(f"Saved pairs to {pairs_path}")

    # Print summary stats
    print("\n=== Dataset Summary ===")
    print(f"Total trades: {len(dataset)}")
    print(f"Total pairs: {len(pairs_df)}")
    print(f"Profit range: {dataset['profit'].min():.0f} to {dataset['profit'].max():.0f}")
    print(f"Profitable trades: {(dataset['profit'] > 0).sum()} ({(dataset['profit'] > 0).mean()*100:.1f}%)")
    print(f"Feature columns: {len(dataset.columns)}")

    conn.close()
    print("\nDataset build complete!")


if __name__ == '__main__':
    main()
