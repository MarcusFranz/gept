"""
Training Tier Configuration
===========================

Centralized configuration for model quality tiers used in the training pipeline.
Tiers are based on AUC (Area Under ROC Curve) and determine training frequency.

Tier System:
- Tier A (AUC >= 0.58): Daily training - highest quality models
- Tier B (AUC >= 0.54): Every 3 days - moderate signal
- Tier C (AUC >= 0.52): Weekly - marginal but viable
- Tier D (AUC < 0.52): Monthly only - discovery/re-evaluation

See cloud/TIERED_TRAINING.md for full documentation of the tiered training system.

NOTE: This is separate from the data quality tiers in src/item_analyzer.py,
which uses numeric tiers (1, 2, 3) based on data completeness metrics.
"""

from typing import Dict


# AUC thresholds for tier classification
# Models with higher AUC get trained more frequently
TIER_THRESHOLDS: Dict[str, float] = {
    'A': 0.58,  # Daily training - high-value predictable items
    'B': 0.54,  # Every 3 days - moderate signal
    'C': 0.52,  # Weekly - marginal but viable
    # D: below 0.52 - monthly discovery only
}

# Training frequency by tier (days between training runs)
TIER_INTERVALS: Dict[str, int] = {
    'A': 1,   # Daily
    'B': 3,   # Every 3 days
    'C': 7,   # Weekly
    'D': 30,  # Monthly (discovery runs only)
}

# Hysteresis: consecutive runs below threshold required for demotion
TIER_DEMOTION_HYSTERESIS: int = 2


def get_tier_from_auc(auc: float) -> str:
    """
    Determine training tier based on average AUC score.

    Args:
        auc: Average AUC score across model targets (0.0 to 1.0)

    Returns:
        Tier letter ('A', 'B', 'C', or 'D')

    Example:
        >>> get_tier_from_auc(0.62)
        'A'
        >>> get_tier_from_auc(0.55)
        'B'
        >>> get_tier_from_auc(0.50)
        'D'
    """
    if auc >= TIER_THRESHOLDS['A']:
        return 'A'
    elif auc >= TIER_THRESHOLDS['B']:
        return 'B'
    elif auc >= TIER_THRESHOLDS['C']:
        return 'C'
    else:
        return 'D'


def get_training_interval(tier: str) -> int:
    """
    Get the training interval in days for a given tier.

    Args:
        tier: Tier letter ('A', 'B', 'C', or 'D')

    Returns:
        Number of days between training runs

    Example:
        >>> get_training_interval('A')
        1
        >>> get_training_interval('D')
        30
    """
    return TIER_INTERVALS.get(tier, TIER_INTERVALS['D'])
