"""Batch comparison of CatBoost vs PatchTST on multiple items."""

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
    config = ModelConfig(n_items=150)
    model = PatchTSTModel(config)
    checkpoint_path = Path("/home/ubuntu/gept/models/patchtst_v1/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded PatchTST from epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', 0):.4f}")
    else:
        return None
    model.train(False)
    return model


def get_item_names():
    """Get item names from database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT item_id, name FROM items")
            return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        release_connection(conn)


def get_actual_prices(item_id, start_time, hours_ahead):
    conn = get_connection()
    try:
        results = {}
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


def test_item(model, item_id, item_idx, test_time, item_name=""):
    """Test a single item and return calibration results."""
    horizons = [1, 2, 4, 8, 12, 24]

    # Get actual prices
    actual = get_actual_prices(item_id, test_time, horizons)
    current = get_actual_prices(item_id, test_time, [0])

    if not actual or not current or 0 not in current:
        return None

    # Load dataset for this item
    dataset = MultiResolutionDataset(
        item_ids=[item_id],
        start_date=datetime(2025, 12, 1),
        end_date=datetime(2026, 1, 18),
        sample_interval_hours=1
    )

    if len(dataset) == 0:
        return None

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

    # Prepare and normalize batch
    batch = {
        "recent_5m": torch.tensor(sample['recent_5m']).unsqueeze(0),
        "medium_1h": torch.tensor(sample['medium_1h']).unsqueeze(0),
        "long_4h": torch.tensor(sample['long_4h']).unsqueeze(0),
        "item_id": torch.tensor([item_idx]),
        "current_high": torch.tensor([sample['current_high']]),
        "current_low": torch.tensor([sample['current_low']]),
    }

    current_mid = (batch['current_high'] + batch['current_low']) / 2
    current_mid = torch.where(current_mid < 1e-8, torch.ones_like(current_mid), current_mid)
    price_norm = current_mid.unsqueeze(-1).unsqueeze(-1)

    # Normalize recent
    recent = batch['recent_5m'].clone()
    recent[:, :, 0:2] = recent[:, :, 0:2] / price_norm
    recent[:, :, 2:4] = torch.log1p(torch.clamp(recent[:, :, 2:4], min=0, max=1e9))
    recent[:, :, 4:5] = recent[:, :, 4:5] / price_norm
    recent[:, :, 5] = torch.clamp(recent[:, :, 5], min=-1, max=1)
    batch['recent_5m'] = recent

    # Normalize medium
    medium = batch['medium_1h'].clone()
    medium[:, :, 0:2] = medium[:, :, 0:2] / price_norm
    medium[:, :, 2:4] = torch.log1p(torch.clamp(medium[:, :, 2:4], min=0, max=1e9))
    medium[:, :, 4:5] = medium[:, :, 4:5] / price_norm
    medium[:, :, 5] = torch.clamp(medium[:, :, 5] / 100, min=0, max=1)
    medium[:, :, 6] = torch.clamp(medium[:, :, 6], min=-1, max=1)
    medium[:, :, 7:9] = medium[:, :, 7:9] / price_norm
    medium[:, :, 9] = torch.log1p(torch.clamp(medium[:, :, 9], min=0, max=1e9))
    batch['medium_1h'] = medium

    # Normalize long
    long_term = batch['long_4h'].clone()
    long_term[:, :, 0:2] = long_term[:, :, 0:2] / price_norm
    long_term[:, :, 2:4] = torch.log1p(torch.clamp(long_term[:, :, 2:4], min=0, max=1e9))
    long_term[:, :, 4:5] = long_term[:, :, 4:5] / price_norm
    long_term[:, :, 5] = torch.clamp(long_term[:, :, 5] / 100, min=0, max=1)
    long_term[:, :, 6] = torch.clamp(long_term[:, :, 6], min=-1, max=1)
    long_term[:, :, 7:9] = long_term[:, :, 7:9] / price_norm
    long_term[:, :, 9] = torch.log1p(torch.clamp(long_term[:, :, 9], min=0, max=1e9))
    batch['long_4h'] = long_term

    # Run inference
    with torch.no_grad():
        output = model(
            recent_5m=batch['recent_5m'],
            medium_1h=batch['medium_1h'],
            long_4h=batch['long_4h'],
            item_ids=batch['item_id'],
            current_high=None,
            current_low=None
        )

    mid_price = current_mid.item()
    high_q = output['high_quantiles'][0].numpy() * mid_price
    low_q = output['low_quantiles'][0].numpy() * mid_price

    patchtst_horizons = [1, 2, 4, 8, 12, 24, 48]

    # Calculate calibration
    results = {
        "item_id": item_id,
        "item_name": item_name,
        "current_price": mid_price,
        "horizons": {}
    }

    for i, h in enumerate(patchtst_horizons):
        if h not in actual:
            continue

        actual_high = actual[h]['high']
        actual_low = actual[h]['low']

        median_high = high_q[i, 2]
        median_low = low_q[i, 2]

        high_err = (actual_high - median_high) / actual_high * 100
        low_err = (actual_low - median_low) / actual_low * 100

        high_in_range = low_q[i, 0] <= actual_high <= high_q[i, 4]
        low_in_range = low_q[i, 0] <= actual_low <= high_q[i, 4]

        results["horizons"][h] = {
            "actual_high": actual_high,
            "actual_low": actual_low,
            "pred_high_median": median_high,
            "pred_low_median": median_low,
            "pred_high_range": (high_q[i, 0], high_q[i, 4]),
            "pred_low_range": (low_q[i, 0], low_q[i, 4]),
            "high_err_pct": high_err,
            "low_err_pct": low_err,
            "high_calibrated": high_in_range,
            "low_calibrated": low_in_range,
        }

    return results


def main():
    print("=" * 70)
    print("Batch Model Comparison: PatchTST Calibration Test")
    print("=" * 70)

    # Load model
    model = load_patchtst()
    if model is None:
        print("Failed to load model!")
        return

    # Get item names
    item_names = get_item_names()

    # Test items from training set (top coverage items)
    test_items = [
        (12924, 0),   # Old school bond
        (2, 1),       # Cannonball
        (562, 2),     # Chaos rune
        (892, 3),     # Rune platebody
        (561, 4),     # Nature rune
        (554, 5),     # Fire rune
        (225, 6),     # Limpwurt root
        (440, 7),     # Coal
        (556, 8),     # Air rune
        (314, 9),     # Feather
        (555, 10),    # Water rune
        (379, 11),    # Lobster
        (564, 12),    # Cosmic rune
        (890, 13),    # Rune platelegs
        (557, 14),    # Earth rune
    ]

    test_time = pd.Timestamp("2026-01-15 12:00:00", tz="UTC")
    print(f"\nTest time: {test_time}")
    print(f"Testing {len(test_items)} items...\n")

    all_results = []

    for item_id, item_idx in test_items:
        item_name = item_names.get(item_id, f"Item {item_id}")
        print(f"Testing {item_name} ({item_id})...", end=" ", flush=True)

        result = test_item(model, item_id, item_idx, test_time, item_name)

        if result is None:
            print("SKIPPED (no data)")
            continue

        all_results.append(result)

        # Quick summary
        calibrated = sum(1 for h, r in result["horizons"].items()
                        if r["high_calibrated"] and r["low_calibrated"])
        total = len(result["horizons"]) * 2
        high_cal = sum(1 for r in result["horizons"].values() if r["high_calibrated"])
        low_cal = sum(1 for r in result["horizons"].values() if r["low_calibrated"])

        print(f"HIGH: {high_cal}/{len(result['horizons'])}, LOW: {low_cal}/{len(result['horizons'])}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_high_cal = 0
    total_low_cal = 0
    total_horizons = 0
    all_high_errors = []
    all_low_errors = []

    for result in all_results:
        for h, r in result["horizons"].items():
            total_horizons += 1
            if r["high_calibrated"]:
                total_high_cal += 1
            if r["low_calibrated"]:
                total_low_cal += 1
            all_high_errors.append(abs(r["high_err_pct"]))
            all_low_errors.append(abs(r["low_err_pct"]))

    print(f"\nItems tested: {len(all_results)}")
    print(f"Total horizon predictions: {total_horizons}")
    print(f"\nCalibration (actual within p10-p90):")
    print(f"  HIGH: {total_high_cal}/{total_horizons} ({100*total_high_cal/total_horizons:.1f}%)")
    print(f"  LOW:  {total_low_cal}/{total_horizons} ({100*total_low_cal/total_horizons:.1f}%)")
    print(f"\nMedian absolute error:")
    print(f"  HIGH: {np.median(all_high_errors):.2f}%")
    print(f"  LOW:  {np.median(all_low_errors):.2f}%")
    print(f"\nMean absolute error:")
    print(f"  HIGH: {np.mean(all_high_errors):.2f}%")
    print(f"  LOW:  {np.mean(all_low_errors):.2f}%")

    # Per-horizon breakdown
    print("\n" + "-" * 70)
    print("Per-Horizon Calibration:")
    print("-" * 70)

    for h in [1, 2, 4, 8, 12, 24]:
        h_high_cal = sum(1 for r in all_results for hr, hv in r["horizons"].items()
                        if hr == h and hv["high_calibrated"])
        h_low_cal = sum(1 for r in all_results for hr, hv in r["horizons"].items()
                       if hr == h and hv["low_calibrated"])
        h_count = sum(1 for r in all_results if h in r["horizons"])

        h_high_err = [abs(r["horizons"][h]["high_err_pct"]) for r in all_results if h in r["horizons"]]
        h_low_err = [abs(r["horizons"][h]["low_err_pct"]) for r in all_results if h in r["horizons"]]

        if h_count > 0:
            print(f"+{h:2d}h: HIGH {h_high_cal}/{h_count} ({100*h_high_cal/h_count:.0f}%), "
                  f"LOW {h_low_cal}/{h_count} ({100*h_low_cal/h_count:.0f}%), "
                  f"median_err: HIGH {np.median(h_high_err):.1f}%, LOW {np.median(h_low_err):.1f}%")


if __name__ == "__main__":
    main()
