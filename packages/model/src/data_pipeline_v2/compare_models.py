"""Compare CatBoost vs PatchTST predictions on historical data."""

import sys
sys.path.insert(0, "/home/ubuntu/gept/packages/model/src")
sys.path.insert(0, "/home/ubuntu/gept/packages/model/src/data_pipeline_v2")
sys.path.insert(0, "/home/ubuntu/gept/src")

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from patchtst_model import PatchTSTModel, ModelConfig
from training_dataset import MultiResolutionDataset
from db_utils import get_connection, release_connection


def load_patchtst():
    config = ModelConfig(n_items=150)  # Match training config
    model = PatchTSTModel(config)

    checkpoint_path = Path("/home/ubuntu/gept/models/patchtst_v1/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded PatchTST from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Val loss: {checkpoint.get('val_loss', 0):.4f}")
    else:
        print("ERROR: No PatchTST checkpoint found!")
        return None

    model.return_value = None
    model.train(False)
    return model


def get_actual_prices(item_id, start_time, hours_ahead):
    """Get actual prices at future time points."""
    conn = get_connection()
    try:
        results = {}
        # Convert to datetime if needed
        if isinstance(start_time, pd.Timestamp):
            start_time = start_time.to_pydatetime()
        for h in hours_ahead:
            target_time = start_time + timedelta(hours=h)
            query = """
                SELECT avg_high_price, avg_low_price
                FROM price_data_5min
                WHERE item_id = %s
                AND timestamp >= %s - INTERVAL '5 minutes'
                AND timestamp <= %s + INTERVAL '5 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                LIMIT 1
            """
            with conn.cursor() as cur:
                cur.execute(query, (item_id, target_time, target_time, target_time))
                row = cur.fetchone()
                if row and row[0] and row[1]:
                    results[h] = {"high": float(row[0]), "low": float(row[1])}
        return results
    finally:
        release_connection(conn)


def get_catboost_predictions(item_id, pred_time):
    """Get CatBoost predictions from the predictions table."""
    conn = get_connection()
    try:
        query = """
            SELECT hour_offset, offset_pct, fill_probability, buy_price, sell_price
            FROM predictions
            WHERE item_id = %s
            AND time >= %s - INTERVAL '10 minutes'
            AND time <= %s + INTERVAL '10 minutes'
            ORDER BY ABS(EXTRACT(EPOCH FROM (time - %s)))
        """
        with conn.cursor() as cur:
            cur.execute(query, (item_id, pred_time, pred_time, pred_time))
            rows = cur.fetchall()

        if not rows:
            return None

        preds = {}
        for row in rows:
            hour_offset, offset_pct, fill_prob, buy_price, sell_price = row
            if hour_offset not in preds:
                preds[hour_offset] = []
            preds[hour_offset].append({
                "offset_pct": float(offset_pct),
                "fill_prob": float(fill_prob) if fill_prob else 0,
                "buy_price": float(buy_price) if buy_price else 0,
                "sell_price": float(sell_price) if sell_price else 0,
            })
        return preds
    finally:
        release_connection(conn)


def main():
    print("=" * 60)
    print("Model Comparison: CatBoost vs PatchTST")
    print("=" * 60)

    item_id = 562  # Chaos rune
    test_time = pd.Timestamp("2026-01-15 12:00:00", tz="UTC")

    print(f"\nTest Item: {item_id} (Chaos rune)")
    print(f"Prediction Time: {test_time}")

    horizons = [1, 2, 4, 8, 12, 24]

    # 1. Get actual prices
    print("\n" + "-" * 40)
    print("1. ACTUAL PRICES (Ground Truth)")
    print("-" * 40)

    actual = get_actual_prices(item_id, test_time, horizons)
    current = get_actual_prices(item_id, test_time, [0])
    if 0 in current:
        print(f"Current: high={current[0]['high']:,.0f}, low={current[0]['low']:,.0f}")

    for h in horizons:
        if h in actual:
            print(f"  +{h:2d}h: high={actual[h]['high']:,.0f}, low={actual[h]['low']:,.0f}")

    # 2. Get CatBoost predictions
    print("\n" + "-" * 40)
    print("2. CATBOOST PREDICTIONS")
    print("-" * 40)

    catboost_preds = get_catboost_predictions(item_id, test_time)
    if catboost_preds:
        for h in horizons:
            if h in catboost_preds:
                preds = catboost_preds[h]
                best = max(preds, key=lambda x: x['fill_prob'])
                print(f"  +{h:2d}h: buy={best['buy_price']:,.0f}, sell={best['sell_price']:,.0f}, fill_prob={best['fill_prob']:.2%}")
    else:
        print("  No CatBoost predictions found for this time")

    # 3. Get PatchTST predictions
    print("\n" + "-" * 40)
    print("3. PATCHTST PREDICTIONS")
    print("-" * 40)

    model = load_patchtst()
    if model is None:
        return

    # Need 30+ days of history before the sample point
    dataset = MultiResolutionDataset(
        item_ids=[item_id],
        start_date=datetime(2025, 12, 1),  # 30+ days before Jan 15
        end_date=datetime(2026, 1, 18),    # 48h after Jan 15
        sample_interval_hours=1
    )

    # Find sample closest to test_time
    best_idx = 0
    best_diff = float("inf")
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_time = pd.Timestamp(sample['timestamp'])
        diff = abs((sample_time - pd.Timestamp(test_time)).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    sample = dataset[best_idx]
    print(f"  Using sample from: {sample['timestamp']}")

    # Prepare batch
    batch = {
        "recent_5m": torch.tensor(sample['recent_5m']).unsqueeze(0),
        "medium_1h": torch.tensor(sample['medium_1h']).unsqueeze(0),
        "long_4h": torch.tensor(sample['long_4h']).unsqueeze(0),
        "item_id": torch.tensor([0]),
        "current_high": torch.tensor([sample['current_high']]),
        "current_low": torch.tensor([sample['current_low']]),
    }

    # Normalize batch - must match training exactly
    current_mid = (batch['current_high'] + batch['current_low']) / 2
    current_mid = torch.where(current_mid < 1e-8, torch.ones_like(current_mid), current_mid)
    price_norm = current_mid.unsqueeze(-1).unsqueeze(-1)

    # Recent 5m: [high, low, high_vol, low_vol, spread, staleness]
    recent = batch['recent_5m'].clone()
    recent[:, :, 0:2] = recent[:, :, 0:2] / price_norm  # Prices -> ratios
    recent[:, :, 2:4] = torch.log1p(torch.clamp(recent[:, :, 2:4], min=0, max=1e9))  # Volume -> log
    recent[:, :, 4:5] = recent[:, :, 4:5] / price_norm  # Spread -> ratio
    recent[:, :, 5] = torch.clamp(recent[:, :, 5], min=-1, max=1)  # Staleness
    batch['recent_5m'] = recent

    # Medium 1h: [high, low, high_vol, low_vol, spread, sample_count, staleness, high_range, low_range, total_vol]
    medium = batch['medium_1h'].clone()
    medium[:, :, 0:2] = medium[:, :, 0:2] / price_norm  # Prices
    medium[:, :, 2:4] = torch.log1p(torch.clamp(medium[:, :, 2:4], min=0, max=1e9))  # Volumes
    medium[:, :, 4:5] = medium[:, :, 4:5] / price_norm  # Spread
    medium[:, :, 5] = torch.clamp(medium[:, :, 5] / 100, min=0, max=1)  # Sample count
    medium[:, :, 6] = torch.clamp(medium[:, :, 6], min=-1, max=1)  # Staleness
    medium[:, :, 7:9] = medium[:, :, 7:9] / price_norm  # Price ranges
    medium[:, :, 9] = torch.log1p(torch.clamp(medium[:, :, 9], min=0, max=1e9))  # Total volume
    batch['medium_1h'] = medium

    # Long 4h: same structure as medium
    long_term = batch['long_4h'].clone()
    long_term[:, :, 0:2] = long_term[:, :, 0:2] / price_norm
    long_term[:, :, 2:4] = torch.log1p(torch.clamp(long_term[:, :, 2:4], min=0, max=1e9))
    long_term[:, :, 4:5] = long_term[:, :, 4:5] / price_norm
    long_term[:, :, 5] = torch.clamp(long_term[:, :, 5] / 100, min=0, max=1)
    long_term[:, :, 6] = torch.clamp(long_term[:, :, 6], min=-1, max=1)
    long_term[:, :, 7:9] = long_term[:, :, 7:9] / price_norm
    long_term[:, :, 9] = torch.log1p(torch.clamp(long_term[:, :, 9], min=0, max=1e9))
    batch['long_4h'] = long_term

    # Run inference - model outputs price RATIOS when not given current_high/low
    # This matches training where targets are normalized by current_mid
    with torch.no_grad():
        output = model(
            recent_5m=batch['recent_5m'],
            medium_1h=batch['medium_1h'],
            long_4h=batch['long_4h'],
            item_ids=batch['item_id'],
            current_high=None,  # Don't scale internally
            current_low=None
        )

    # Model outputs ratios - multiply by current price to get actual prices
    mid_price = current_mid.item()
    high_q = output['high_quantiles'][0].numpy() * mid_price
    low_q = output['low_quantiles'][0].numpy() * mid_price

    patchtst_horizons = [1, 2, 4, 8, 12, 24, 48]
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n  Quantile predictions (p10, p30, p50, p70, p90):")
    for i, h in enumerate(patchtst_horizons):
        if h in horizons or h == 48:
            high_str = " ".join([f"{high_q[i, j]:,.0f}" for j in range(5)])
            low_str = " ".join([f"{low_q[i, j]:,.0f}" for j in range(5)])
            print(f"  +{h:2d}h HIGH: [{high_str}]")
            print(f"       LOW:  [{low_str}]")

    # 4. Compare accuracy
    print("\n" + "-" * 40)
    print("4. ACCURACY COMPARISON")
    print("-" * 40)

    print("\nPatchTST Calibration (did actual fall within p10-p90 range?):")
    for i, h in enumerate(patchtst_horizons):
        if h not in actual:
            continue

        actual_high = actual[h]['high']
        actual_low = actual[h]['low']

        median_high_err = (actual_high - high_q[i, 2]) / actual_high * 100
        median_low_err = (actual_low - low_q[i, 2]) / actual_low * 100

        high_in_range = low_q[i, 0] <= actual_high <= high_q[i, 4]
        low_in_range = low_q[i, 0] <= actual_low <= high_q[i, 4]

        status_high = "Y" if high_in_range else "N"
        status_low = "Y" if low_in_range else "N"

        print(f"  +{h:2d}h: HIGH {status_high} (median err={median_high_err:+.1f}%)")
        print(f"       LOW  {status_low} (median err={median_low_err:+.1f}%)")


if __name__ == "__main__":
    main()
