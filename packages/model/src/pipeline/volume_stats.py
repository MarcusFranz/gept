"""
Volume Statistics Computation
=============================

Computes per-item mean/std for log-transformed volume targets.
Items with fewer than min_samples fall back to global statistics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict

MIN_SAMPLES_DEFAULT = 100
EPSILON = 1e-6


def compute_volume_stats(
    volumes_by_item: Dict[int, np.ndarray],
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> Dict:
    """
    Compute per-item statistics for volume normalization.

    Args:
        volumes_by_item: {item_id: array of raw volume values}
        min_samples: Minimum samples for item-specific stats

    Returns:
        Dict with per-item stats and _metadata key
    """
    all_log_volumes = []
    for item_id, vols in volumes_by_item.items():
        all_log_volumes.extend(np.log1p(vols.astype(np.float64)).tolist())

    all_log_volumes = np.array(all_log_volumes)
    global_mean = float(np.mean(all_log_volumes))
    global_std = float(max(np.std(all_log_volumes), EPSILON))

    stats = {}
    item_sample_counts = {}

    for item_id, vols in volumes_by_item.items():
        log_vols = np.log1p(vols.astype(np.float64))
        item_sample_counts[item_id] = len(vols)

        if len(vols) >= min_samples:
            item_mean = float(np.mean(log_vols))
            item_std = float(max(np.std(log_vols), EPSILON))
            stats[item_id] = {'mean': item_mean, 'std': item_std}
        else:
            stats[item_id] = {'mean': global_mean, 'std': global_std, 'fallback': True}

    stats['_metadata'] = {
        'min_samples': min_samples,
        'transform': 'log1p',
        'epsilon': EPSILON,
        'global_mean': global_mean,
        'global_std': global_std,
        'total_items': len(volumes_by_item),
        'fallback_items': sum(1 for s in stats.values() if isinstance(s, dict) and s.get('fallback')),
    }

    return stats


def normalize_volume(raw_volume: np.ndarray, item_id: int, stats: Dict) -> np.ndarray:
    """Normalize raw volume to z-score using precomputed stats."""
    item_stats = stats.get(item_id, stats.get('_metadata', {}))

    if isinstance(item_stats, dict) and 'mean' in item_stats:
        mean = item_stats['mean']
        std = item_stats['std']
    else:
        meta = stats.get('_metadata', {})
        mean = meta.get('global_mean', 0.0)
        std = meta.get('global_std', 1.0)

    log_volume = np.log1p(raw_volume.astype(np.float64))
    return (log_volume - mean) / std


def save_volume_stats(stats: Dict, path: Path) -> None:
    """Save stats to JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    with open(path, 'w') as f:
        json.dump(convert(stats), f, indent=2)


def load_volume_stats(path: Path) -> Dict:
    """Load stats from JSON file."""
    with open(path) as f:
        stats = json.load(f)
    result = {}
    for k, v in stats.items():
        if k == '_metadata':
            result[k] = v
        else:
            try:
                result[int(k)] = v
            except ValueError:
                result[k] = v
    return result
