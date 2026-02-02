"""
Calibration Layer for Multi-Target Probability Predictions
==========================================================

Implements isotonic regression calibration to correct systematic under-prediction
in model probability outputs. Based on calibration analysis showing models are
3.76x under-predicted in 0-30% range, with Hour 1 being 9x under-predicted.

Architecture:
- 11 hour-bucket groups (pools offsets within each bucket)
- Isotonic regression per group (handles non-linear miscalibration)
- Calibrated values capped at CALIBRATED_MAX (from inference_config)

Usage:
    # Load calibration for a model
    calibrator = CalibrationManager.load('models/20260111_142024/2/calibration.json')

    # Apply calibration
    calibrated_prob = calibrator.calibrate(hour=1, raw_prob=0.05)
"""

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Import calibration bounds from inference_config (single source of truth)
from inference_config import CALIBRATED_MAX, CALIBRATED_MIN


@dataclass
class CalibrationConfig:
    """Configuration for hour-based calibration grouping."""

    # Hour group mappings: group_name -> list of hours in that group
    hour_groups: Dict[str, List[int]] = field(default_factory=lambda: {
        'hour_1': [1],
        'hour_2': [2],
        'hour_3': [3],
        'hour_4': [4],
        'hours_5_6': [5, 6],
        'hours_7_8': [7, 8],
        'hours_9_10': [9, 10],
        'hours_11_12': [11, 12],
        'hours_16_20': [16, 20],
        'hours_24_32': [24, 32],
        'hours_40_48': [40, 48],
    })

    @classmethod
    def get_hour_group(cls, hour: int) -> str:
        """Map an hour to its calibration group name."""
        config = cls()
        for group_name, hours in config.hour_groups.items():
            if hour in hours:
                return group_name
        # Fallback for hours not in explicit groups (shouldn't happen with 108 targets)
        return f'hour_{hour}'

    @classmethod
    def get_all_groups(cls) -> List[str]:
        """Return all configured hour group names."""
        config = cls()
        return list(config.hour_groups.keys())


class IsotonicCalibrator:
    """
    Single isotonic regression calibrator for one hour group.

    Stores fitted x/y thresholds and performs interpolation for calibration.
    """

    def __init__(
        self,
        x_thresholds: Optional[List[float]] = None,
        y_thresholds: Optional[List[float]] = None,
        n_samples: int = 0,
        brier_before: Optional[float] = None,
        brier_after: Optional[float] = None
    ):
        """
        Initialize calibrator.

        Args:
            x_thresholds: Input probability thresholds from isotonic fit
            y_thresholds: Output calibrated probability thresholds
            n_samples: Number of samples used to fit
            brier_before: Brier score before calibration
            brier_after: Brier score after calibration
        """
        self.x_thresholds = np.array(x_thresholds) if x_thresholds else np.array([])
        self.y_thresholds = np.array(y_thresholds) if y_thresholds else np.array([])
        self.n_samples = n_samples
        self.brier_before = brier_before
        self.brier_after = brier_after

    @property
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted."""
        return len(self.x_thresholds) > 0 and len(self.y_thresholds) > 0

    def calibrate(self, prob: float) -> float:
        """
        Apply isotonic calibration to a raw probability.

        Uses numpy interp for efficient piecewise linear interpolation
        between the fitted isotonic thresholds.

        Args:
            prob: Raw probability from model

        Returns:
            Calibrated probability, clipped to [CALIBRATED_MIN, CALIBRATED_MAX]
        """
        if not self.is_fitted:
            # No calibration available - return clamped input
            return float(np.clip(prob, CALIBRATED_MIN, CALIBRATED_MAX))

        # Interpolate using fitted isotonic curve
        calibrated = float(np.interp(prob, self.x_thresholds, self.y_thresholds))

        # Enforce bounds (calibration should not boost above CALIBRATED_MAX)
        return float(np.clip(calibrated, CALIBRATED_MIN, CALIBRATED_MAX))

    def to_dict(self) -> Dict:
        """Serialize calibrator to dictionary."""
        return {
            'x_thresholds': self.x_thresholds.tolist() if self.is_fitted else [],
            'y_thresholds': self.y_thresholds.tolist() if self.is_fitted else [],
            'n_samples': self.n_samples,
            'brier_before': self.brier_before,
            'brier_after': self.brier_after
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'IsotonicCalibrator':
        """Deserialize calibrator from dictionary."""
        return cls(
            x_thresholds=data.get('x_thresholds', []),
            y_thresholds=data.get('y_thresholds', []),
            n_samples=data.get('n_samples', 0),
            brier_before=data.get('brier_before'),
            brier_after=data.get('brier_after')
        )

    @classmethod
    def fit(
        cls,
        predictions: np.ndarray,
        actuals: np.ndarray,
        y_min: float = CALIBRATED_MIN,
        y_max: float = CALIBRATED_MAX
    ) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression calibrator from predictions and actual outcomes.

        Args:
            predictions: Array of raw model probabilities
            actuals: Array of actual binary outcomes (0 or 1)
            y_min: Minimum calibrated probability
            y_max: Maximum calibrated probability

        Returns:
            Fitted IsotonicCalibrator instance
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss

        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        if len(predictions) < 10:
            # Not enough data to fit - return empty calibrator
            return cls()

        # Compute Brier score before calibration
        brier_before = brier_score_loss(actuals, predictions)

        # Fit isotonic regression
        iso = IsotonicRegression(
            y_min=y_min,
            y_max=y_max,
            out_of_bounds='clip'
        )
        iso.fit(predictions, actuals)

        # Get calibrated predictions for Brier score after
        calibrated_preds = iso.predict(predictions)
        brier_after = brier_score_loss(actuals, calibrated_preds)

        return cls(
            x_thresholds=iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else [],
            y_thresholds=iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else [],
            n_samples=len(predictions),
            brier_before=float(brier_before),
            brier_after=float(brier_after)
        )


class CalibrationManager:
    """
    Manages calibration for all hour groups of a single item/model.

    Provides hour-based calibration lookup and application.
    """

    def __init__(
        self,
        calibrators: Optional[Dict[str, IsotonicCalibrator]] = None,
        version: str = '1.0',
        fitted_at: Optional[str] = None,
        item_id: Optional[int] = None,
        global_metrics: Optional[Dict] = None
    ):
        """
        Initialize calibration manager.

        Args:
            calibrators: Dict mapping hour group names to IsotonicCalibrator instances
            version: Calibration format version
            fitted_at: ISO timestamp when calibration was fitted
            item_id: Item ID this calibration is for
            global_metrics: Overall calibration metrics (brier improvement, etc.)
        """
        self.calibrators = calibrators or {}
        self.version = version
        self.fitted_at = fitted_at or datetime.now(timezone.utc).isoformat()
        self.item_id = item_id
        self.global_metrics = global_metrics or {}

    def calibrate(self, hour: int, prob: float) -> float:
        """
        Apply calibration to a raw probability for a given hour.

        Args:
            hour: Target hour (1-48)
            prob: Raw probability from model

        Returns:
            Calibrated probability
        """
        group_name = CalibrationConfig.get_hour_group(hour)

        if group_name in self.calibrators:
            return self.calibrators[group_name].calibrate(prob)

        # No calibrator for this group - return clamped input
        return float(np.clip(prob, CALIBRATED_MIN, CALIBRATED_MAX))

    def has_calibration(self, hour: int) -> bool:
        """Check if calibration is available for a given hour."""
        group_name = CalibrationConfig.get_hour_group(hour)
        return group_name in self.calibrators and self.calibrators[group_name].is_fitted

    def get_stats(self) -> Dict:
        """Get summary statistics for all calibrators."""
        stats = {
            'n_groups_fitted': 0,
            'total_samples': 0,
            'avg_brier_before': None,
            'avg_brier_after': None,
            'groups': {}
        }

        brier_before_sum = 0.0
        brier_after_sum = 0.0
        brier_count = 0

        for group_name, calibrator in self.calibrators.items():
            if calibrator.is_fitted:
                stats['n_groups_fitted'] += 1
                stats['total_samples'] += calibrator.n_samples
                stats['groups'][group_name] = {
                    'n_samples': calibrator.n_samples,
                    'brier_before': calibrator.brier_before,
                    'brier_after': calibrator.brier_after
                }

                if calibrator.brier_before is not None and calibrator.brier_after is not None:
                    brier_before_sum += calibrator.brier_before
                    brier_after_sum += calibrator.brier_after
                    brier_count += 1

        if brier_count > 0:
            stats['avg_brier_before'] = brier_before_sum / brier_count
            stats['avg_brier_after'] = brier_after_sum / brier_count

        return stats

    def to_dict(self) -> Dict:
        """Serialize calibration manager to dictionary."""
        return {
            'version': self.version,
            'fitted_at': self.fitted_at,
            'item_id': self.item_id,
            'method': 'isotonic',
            'hour_groups': {
                name: cal.to_dict()
                for name, cal in self.calibrators.items()
            },
            'global_metrics': self.global_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationManager':
        """Deserialize calibration manager from dictionary."""
        calibrators = {
            name: IsotonicCalibrator.from_dict(cal_data)
            for name, cal_data in data.get('hour_groups', {}).items()
        }

        return cls(
            calibrators=calibrators,
            version=data.get('version', '1.0'),
            fitted_at=data.get('fitted_at'),
            item_id=data.get('item_id'),
            global_metrics=data.get('global_metrics', {})
        )

    def save(self, path: str) -> None:
        """Save calibration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Optional['CalibrationManager']:
        """
        Load calibration from JSON file.

        Args:
            path: Path to calibration.json file

        Returns:
            CalibrationManager instance or None if file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)


def compute_brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error for probabilities).

    Lower is better. Perfect calibration = 0.

    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes (0 or 1)

    Returns:
        Brier score
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    return float(np.mean((predictions - actuals) ** 2))


def compute_calibration_error(predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute expected calibration error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes (0 or 1)
        n_bins: Number of probability bins

    Returns:
        Expected calibration error
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(predictions)

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_pred_mean = np.mean(predictions[mask])
            bin_actual_mean = np.mean(actuals[mask])
            ece += (bin_count / total_samples) * abs(bin_pred_mean - bin_actual_mean)

    return float(ece)
