"""Model lifecycle management utilities.

This module provides utilities for managing model lifecycle status,
ensuring recommendations only use ACTIVE models while supporting
existing trades using deprecated models.

Core Principle: "No new trades on deprecated models, but always
honor existing positions until they close naturally"
"""

from enum import Enum


class ModelStatus(str, Enum):
    """Model lifecycle status values."""

    ACTIVE = "ACTIVE"  # Model is current and should be used for new recommendations
    DEPRECATED = "DEPRECATED"  # Model is outdated, don't use for new trades
    SUNSET = "SUNSET"  # Model being phased out
    ARCHIVED = "ARCHIVED"  # Model no longer in use


def should_include_for_recommendations(status: str) -> bool:
    """Check if a model should be used for new recommendations.

    Only ACTIVE models should be used for generating new trade recommendations.

    Args:
        status: Model status string

    Returns:
        True if the model should be included in new recommendations
    """
    return status == ModelStatus.ACTIVE


def can_serve_prediction(status: str, is_existing_trade: bool = False) -> bool:
    """Check if a model can serve predictions.

    Existing trades can use any model (even deprecated ones) to honor
    positions until they close naturally. New recommendations only use
    ACTIVE models.

    Args:
        status: Model status string
        is_existing_trade: Whether this is for an existing trade lookup

    Returns:
        True if the model can serve predictions for this use case
    """
    if is_existing_trade:
        # Existing trades can use any model to honor open positions
        return True
    return status == ModelStatus.ACTIVE


def is_active_status(status: str) -> bool:
    """Check if status represents an active, usable model.

    Args:
        status: Model status string

    Returns:
        True if status is ACTIVE
    """
    return status == ModelStatus.ACTIVE


def is_deprecated_status(status: str) -> bool:
    """Check if status represents a deprecated model.

    Args:
        status: Model status string

    Returns:
        True if status is DEPRECATED, SUNSET, or ARCHIVED
    """
    return status in (ModelStatus.DEPRECATED, ModelStatus.SUNSET, ModelStatus.ARCHIVED)
