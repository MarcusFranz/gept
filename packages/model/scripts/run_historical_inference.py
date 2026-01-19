#!/usr/bin/env python3
"""Run historical inference to get prediction metadata for each trade.

This script:
1. For each trade with price data coverage
2. Loads price data up to the trade timestamp
3. Computes features at that point in time
4. Runs the production model to get fill_probability, EV, etc.
5. Stores predictions in historical_predictions table
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engine import FeatureEngine, Granularity
from batch_predictor_multitarget import MultiTargetBatchPredictor


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'osrs_data'),
        user=os.getenv('DB_USER', 'osrs_user'),
        password=os.getenv('DB_PASS', 'osrs_price_data_2024'),
    )


def load_trades_for_inference(conn) -> pd.DataFrame:
    """Load trades that need prediction inference."""
    query = """
        SELECT
            t.id as trade_id,
            t.first_buy_time,
            t.item_id,
            t.item_name,
            t.avg_buy_price,
            t.avg_sell_price,
            t.profit
        FROM training_trades t
        LEFT JOIN historical_predictions hp ON hp.training_trade_id = t.id
        WHERE t.status = 'FINISHED'
          AND t.has_price_data = TRUE
          AND t.item_id IS NOT NULL
          AND hp.id IS NULL  -- Not yet processed
        ORDER BY t.first_buy_time
    """
    return pd.read_sql(query, conn)


def load_price_data_for_item(conn, item_id: int, end_time: datetime, hours: int = 168) -> pd.DataFrame:
    """Load price data for an item up to a specific timestamp.

    Args:
        conn: Database connection
        item_id: OSRS item ID
        end_time: Load data up to this time
        hours: Number of hours of history to load (default 168 = 1 week)
    """
    start_time = end_time - timedelta(hours=hours)

    cur = conn.cursor()
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
    """, (item_id, start_time, end_time))

    rows = cur.fetchall()
    cur.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'timestamp', 'avg_high_price', 'avg_low_price',
        'high_price_volume', 'low_price_volume'
    ])
    return df


def run_inference_for_trade(
    predictor: MultiTargetBatchPredictor,
    feature_engine: FeatureEngine,
    conn,
    trade: pd.Series,
) -> List[Dict]:
    """Run model inference for a single trade.

    Returns list of predictions (one per hour_offset/offset_pct combination).
    """
    item_id = trade['item_id']
    trade_time = trade['first_buy_time']

    # Check if model exists for this item
    if item_id not in predictor.models:
        return []

    # Load price data up to trade time
    price_df = load_price_data_for_item(conn, item_id, trade_time, hours=168)

    if len(price_df) < 300:  # Need enough data for feature computation
        return []

    # Compute features
    features_df = feature_engine.compute_features(price_df)

    if len(features_df) < 300:
        return []

    # Run prediction
    try:
        predictions, _, _, _ = predictor.predict_item(
            item_id=item_id,
            features_df=features_df,
            prediction_time=trade_time,
            hour_start=1,
            hour_end=48
        )
    except Exception as e:
        print(f"Error predicting item {item_id}: {e}")
        return []

    return predictions


def store_predictions(conn, trade_id: int, item_id: int, trade_time: datetime, predictions: List[Dict]):
    """Store predictions in historical_predictions table."""
    if not predictions:
        return

    cur = conn.cursor()

    values = []
    for pred in predictions:
        values.append((
            trade_id,
            item_id,
            trade_time,
            'current',  # model_month - using current model
            pred.get('hour_offset', 0),
            pred.get('offset_pct', 0.02),
            pred.get('fill_probability'),
            pred.get('expected_value'),
            pred.get('buy_price'),
            pred.get('sell_price'),
            pred.get('confidence'),
        ))

    insert_sql = """
        INSERT INTO historical_predictions (
            training_trade_id, item_id, prediction_time, model_month,
            hour_offset, offset_pct, fill_probability, expected_value,
            buy_price, sell_price, confidence
        ) VALUES %s
        ON CONFLICT (training_trade_id, hour_offset, offset_pct) DO NOTHING
    """

    execute_values(cur, insert_sql, values)
    conn.commit()
    cur.close()


def main():
    print("Starting historical inference...")

    conn = get_db_connection()

    # Load trades needing inference
    print("Loading trades...")
    trades_df = load_trades_for_inference(conn)
    print(f"Found {len(trades_df)} trades needing inference")

    if len(trades_df) == 0:
        print("No trades to process")
        return

    # Initialize predictor with latest models
    print("Loading models...")
    predictor = MultiTargetBatchPredictor(models_dir=None)  # Auto-detect latest
    print(f"Loaded models for {len(predictor.models)} items")

    # Initialize feature engine
    feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)

    # Process each trade
    success_count = 0
    fail_count = 0
    no_model_count = 0

    for _, trade in tqdm(trades_df.iterrows(), total=len(trades_df), desc="Running inference"):
        item_id = trade['item_id']

        # Check if we have a model for this item
        if item_id not in predictor.models:
            no_model_count += 1
            continue

        # Run inference
        predictions = run_inference_for_trade(predictor, feature_engine, conn, trade)

        if predictions:
            # Store predictions
            store_predictions(
                conn,
                trade['trade_id'],
                item_id,
                trade['first_buy_time'],
                predictions
            )
            success_count += 1
        else:
            fail_count += 1

    print(f"\n=== Inference Complete ===")
    print(f"Success: {success_count}")
    print(f"Failed (no data): {fail_count}")
    print(f"No model: {no_model_count}")

    # Verify stored predictions
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM historical_predictions")
    total_predictions = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT training_trade_id) FROM historical_predictions")
    unique_trades = cur.fetchone()[0]
    cur.close()

    print(f"\nStored predictions: {total_predictions}")
    print(f"Unique trades with predictions: {unique_trades}")

    conn.close()


if __name__ == '__main__':
    main()
