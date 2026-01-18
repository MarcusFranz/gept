"""
Inference Configuration Constants
=================================

Centralized configuration for inference-time thresholds and bounds.
These values are validated against calibration analysis and should
not be changed without re-running validation experiments.

See docs/calibration_analysis.md for the analysis that derived these thresholds.
"""

from typing import Tuple

# Probability bounds (from calibration_analysis.md)
# Predictions outside these bounds indicate model calibration failures
MAX_PROBABILITY = 0.30  # 30% - predictions above this are broken
MIN_PROBABILITY = 0.001  # 0.1% - minimum meaningful probability

# Calibrated probability bounds
# Calibration layer uses a lower max to leave headroom below MAX_PROBABILITY
CALIBRATED_MAX = 0.25  # 25% - calibration output cap (below MAX_PROBABILITY)
CALIBRATED_MIN = MIN_PROBABILITY  # Same minimum


def clip_probability(prob: float) -> Tuple[float, bool]:
    """
    Clip probability to valid bounds.

    Args:
        prob: Raw probability value from model prediction

    Returns:
        tuple: (clipped_value, was_clipped)
            - clipped_value: Probability clamped to [MIN_PROBABILITY, MAX_PROBABILITY]
            - was_clipped: True if the value was modified, False otherwise
    """
    if prob > MAX_PROBABILITY:
        return MAX_PROBABILITY, True
    elif prob < MIN_PROBABILITY:
        return MIN_PROBABILITY, True
    return prob, False


# AUC confidence thresholds (Issue #70)
# These thresholds determine prediction confidence based on model AUC performance.
# Derived from historical model performance analysis.
# - HIGH: Models with strong discriminative ability (AUC >= 0.75)
# - MEDIUM: Models with moderate discriminative ability (AUC >= 0.65)
# - LOW: Models with weak discriminative ability (AUC < 0.65)
HIGH_AUC_THRESHOLD = 0.75  # AUC >= 0.75 for high confidence
MEDIUM_AUC_THRESHOLD = 0.65  # AUC >= 0.65 for medium confidence
LOW_AUC_THRESHOLD = 0.55  # AUC >= 0.55 for low confidence (vs very_low)
DEFAULT_AUC = 0.5  # Default AUC when model performance unknown (random baseline)

# Brier score thresholds for confidence tiers (used in predictor.py)
# Lower Brier scores indicate better calibration
HIGH_BRIER_THRESHOLD = 0.15  # Brier <= 0.15 for high confidence
MEDIUM_BRIER_THRESHOLD = 0.25  # Brier <= 0.25 for medium confidence


def get_confidence_tier(auc: float, default_auc: float = DEFAULT_AUC) -> str:
    """
    Get confidence tier based on AUC score.

    Args:
        auc: Model AUC score
        default_auc: Default AUC if not provided

    Returns:
        Confidence tier: 'high', 'medium', or 'low'
    """
    if auc >= HIGH_AUC_THRESHOLD:
        return 'high'
    elif auc >= MEDIUM_AUC_THRESHOLD:
        return 'medium'
    else:
        return 'low'


def get_confidence_tier_with_brier(
    roc_auc: float,
    brier_score: float,
    default_auc: float = DEFAULT_AUC
) -> str:
    """
    Get confidence tier based on both AUC and Brier score.

    Used in predictor.py for more granular confidence assessment.

    Args:
        roc_auc: Model ROC-AUC score
        brier_score: Model Brier score (lower is better)
        default_auc: Default AUC if not provided

    Returns:
        Confidence tier: 'high', 'medium', 'low', or 'very_low'
    """
    if roc_auc >= HIGH_AUC_THRESHOLD and brier_score <= HIGH_BRIER_THRESHOLD:
        return 'high'
    elif roc_auc >= MEDIUM_AUC_THRESHOLD and brier_score <= MEDIUM_BRIER_THRESHOLD:
        return 'medium'
    elif roc_auc >= LOW_AUC_THRESHOLD:
        return 'low'
    else:
        return 'very_low'


def clip_calibrated_probability(prob: float) -> Tuple[float, bool]:
    """
    Clip calibrated probability to valid bounds.

    Used after calibration layer to enforce tighter bounds than raw predictions.
    Calibration output is capped at CALIBRATED_MAX (0.25) to leave headroom
    below MAX_PROBABILITY (0.30).

    Args:
        prob: Calibrated probability value

    Returns:
        tuple: (clipped_value, was_clipped)
            - clipped_value: Probability clamped to [CALIBRATED_MIN, CALIBRATED_MAX]
            - was_clipped: True if the value was modified, False otherwise
    """
    if prob > CALIBRATED_MAX:
        return CALIBRATED_MAX, True
    elif prob < CALIBRATED_MIN:
        return CALIBRATED_MIN, True
    return prob, False
