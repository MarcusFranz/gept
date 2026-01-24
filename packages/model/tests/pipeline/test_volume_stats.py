# packages/model/tests/pipeline/test_volume_stats.py
import numpy as np
import pytest
from src.pipeline.volume_stats import compute_volume_stats, normalize_volume


def test_compute_volume_stats_basic():
    """Stats computed correctly from raw volumes."""
    volumes = {
        0: np.array([100, 200, 150, 300, 250]),
        1: np.array([10000, 20000, 15000]),
        2: np.array([5, 10]),  # low sample count
    }

    stats = compute_volume_stats(volumes, min_samples=3)

    assert 0 in stats
    assert 'mean' in stats[0]
    assert 'std' in stats[0]
    assert stats[0]['std'] > 1e-6
    assert stats[0].get('fallback') is None

    assert 1 in stats
    assert stats[1].get('fallback') is None

    assert 2 in stats
    assert stats[2]['fallback'] is True


def test_normalize_volume():
    """Normalization produces z-scores."""
    stats = {0: {'mean': 5.0, 'std': 1.0, 'fallback': False}}
    raw_volume = np.array([100.0])
    normalized = normalize_volume(raw_volume, item_id=0, stats=stats)
    expected = (np.log1p(100.0) - 5.0) / 1.0
    np.testing.assert_allclose(normalized, expected, rtol=1e-5)


def test_stats_metadata():
    """Stats file includes metadata."""
    volumes = {0: np.array([100, 200, 150])}
    stats = compute_volume_stats(volumes, min_samples=2)

    assert '_metadata' in stats
    assert stats['_metadata']['min_samples'] == 2
    assert stats['_metadata']['transform'] == 'log1p'
    assert stats['_metadata']['epsilon'] == 1e-6
    assert 'global_mean' in stats['_metadata']
    assert 'global_std' in stats['_metadata']
