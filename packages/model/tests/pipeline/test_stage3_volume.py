# packages/model/tests/pipeline/test_stage3_volume.py
import numpy as np
import pytest


def test_volume_targets_shape():
    """Volume targets have correct shape (N, 7, 2)."""
    from src.pipeline.features import compute_volume_targets

    n_bars = 1000
    data = {
        'high_price_volume': np.random.randint(100, 10000, n_bars).astype(np.float32),
        'low_price_volume': np.random.randint(100, 10000, n_bars).astype(np.float32),
    }

    horizons = [1, 2, 4, 8, 12, 24, 48]
    current_idx = 100

    targets = compute_volume_targets(data, current_idx, horizons, bars_per_hour=12)

    assert targets.shape == (7, 2)
    assert targets.dtype == np.float64


def test_volume_targets_excludes_current_bar():
    """Volume target window starts at current_idx + 1."""
    from src.pipeline.features import compute_volume_targets

    n_bars = 1000
    data = {
        'high_price_volume': np.zeros(n_bars, dtype=np.float32),
        'low_price_volume': np.zeros(n_bars, dtype=np.float32),
    }

    current_idx = 100
    data['high_price_volume'][current_idx] = 99999.0
    data['low_price_volume'][current_idx] = 99999.0
    data['high_price_volume'][current_idx + 1] = 100.0
    data['low_price_volume'][current_idx + 1] = 50.0

    horizons = [1]
    targets = compute_volume_targets(data, current_idx, horizons, bars_per_hour=12)

    assert targets[0, 0] == 50.0   # buy side
    assert targets[0, 1] == 100.0  # sell side


def test_volume_targets_sums_over_horizon():
    """Volume target is sum over future window."""
    from src.pipeline.features import compute_volume_targets

    n_bars = 1000
    data = {
        'high_price_volume': np.ones(n_bars, dtype=np.float32) * 10,
        'low_price_volume': np.ones(n_bars, dtype=np.float32) * 5,
    }

    current_idx = 100
    horizons = [1, 2]

    targets = compute_volume_targets(data, current_idx, horizons, bars_per_hour=12)

    assert targets[0, 0] == 60.0   # 12 bars * 5
    assert targets[0, 1] == 120.0  # 12 bars * 10
    assert targets[1, 0] == 120.0  # 24 bars * 5
    assert targets[1, 1] == 240.0  # 24 bars * 10
