"""
Unit tests for calibration module.
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration import (
    CalibrationConfig,
    IsotonicCalibrator,
    CalibrationManager,
    CALIBRATED_MAX,
    CALIBRATED_MIN,
    compute_brier_score,
    compute_calibration_error
)


class TestCalibrationConfig:
    """Tests for CalibrationConfig class."""

    def test_get_hour_group_individual_hours(self):
        """Hours 1-4 should map to individual groups."""
        assert CalibrationConfig.get_hour_group(1) == 'hour_1'
        assert CalibrationConfig.get_hour_group(2) == 'hour_2'
        assert CalibrationConfig.get_hour_group(3) == 'hour_3'
        assert CalibrationConfig.get_hour_group(4) == 'hour_4'

    def test_get_hour_group_combined_hours(self):
        """Hours 5+ should map to combined groups."""
        assert CalibrationConfig.get_hour_group(5) == 'hours_5_6'
        assert CalibrationConfig.get_hour_group(6) == 'hours_5_6'
        assert CalibrationConfig.get_hour_group(7) == 'hours_7_8'
        assert CalibrationConfig.get_hour_group(8) == 'hours_7_8'

    def test_get_hour_group_extended_hours(self):
        """Extended hours should map correctly."""
        assert CalibrationConfig.get_hour_group(16) == 'hours_16_20'
        assert CalibrationConfig.get_hour_group(20) == 'hours_16_20'
        assert CalibrationConfig.get_hour_group(24) == 'hours_24_32'
        assert CalibrationConfig.get_hour_group(32) == 'hours_24_32'
        assert CalibrationConfig.get_hour_group(40) == 'hours_40_48'
        assert CalibrationConfig.get_hour_group(48) == 'hours_40_48'

    def test_get_all_groups_returns_list(self):
        """Should return all hour group names."""
        groups = CalibrationConfig.get_all_groups()
        assert isinstance(groups, list)
        assert 'hour_1' in groups
        assert 'hours_5_6' in groups
        assert len(groups) == 11


class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator class."""

    def test_unfitted_calibrator_returns_clamped_input(self):
        """Unfitted calibrator should just clamp the input."""
        calibrator = IsotonicCalibrator()
        assert not calibrator.is_fitted

        # Should clamp to bounds
        assert calibrator.calibrate(0.5) == CALIBRATED_MAX
        assert calibrator.calibrate(0.0001) == CALIBRATED_MIN
        assert calibrator.calibrate(0.10) == 0.10

    def test_fitted_calibrator_interpolates(self):
        """Fitted calibrator should interpolate between thresholds."""
        calibrator = IsotonicCalibrator(
            x_thresholds=[0.0, 0.1, 0.2],
            y_thresholds=[0.0, 0.2, 0.24],
            n_samples=100
        )
        assert calibrator.is_fitted

        # Test interpolation
        assert calibrator.calibrate(0.0) == pytest.approx(0.001, abs=0.001)  # Clamped to min
        assert calibrator.calibrate(0.1) == pytest.approx(0.2, abs=0.001)
        assert calibrator.calibrate(0.2) == pytest.approx(0.24, abs=0.001)
        assert calibrator.calibrate(0.05) == pytest.approx(0.1, abs=0.01)  # Midpoint interpolation

    def test_calibrator_respects_bounds(self):
        """Calibrator should not exceed CALIBRATED_MAX."""
        calibrator = IsotonicCalibrator(
            x_thresholds=[0.0, 0.1],
            y_thresholds=[0.0, 0.5],  # Would map 0.1 to 0.5 without bounds
            n_samples=100
        )

        # Should be capped at CALIBRATED_MAX
        assert calibrator.calibrate(0.1) == CALIBRATED_MAX

    def test_to_dict_and_from_dict(self):
        """Should serialize and deserialize correctly."""
        original = IsotonicCalibrator(
            x_thresholds=[0.0, 0.1, 0.2],
            y_thresholds=[0.0, 0.15, 0.24],
            n_samples=100,
            brier_before=0.05,
            brier_after=0.02
        )

        data = original.to_dict()
        restored = IsotonicCalibrator.from_dict(data)

        assert restored.is_fitted
        assert restored.n_samples == 100
        assert restored.brier_before == 0.05
        assert restored.brier_after == 0.02
        np.testing.assert_array_almost_equal(restored.x_thresholds, original.x_thresholds)
        np.testing.assert_array_almost_equal(restored.y_thresholds, original.y_thresholds)

    def test_fit_with_sample_data(self):
        """Should fit isotonic regression from predictions and actuals."""
        # Simulate under-prediction scenario (model predicts low, actuals are higher)
        np.random.seed(42)
        n = 500
        predictions = np.random.uniform(0.01, 0.15, n)
        # Actuals are higher than predictions (under-prediction)
        actuals = np.random.binomial(1, predictions * 2, n)

        calibrator = IsotonicCalibrator.fit(predictions, actuals)

        assert calibrator.is_fitted
        assert calibrator.n_samples == n
        assert calibrator.brier_before is not None
        assert calibrator.brier_after is not None
        # Calibration should improve Brier score
        assert calibrator.brier_after <= calibrator.brier_before

    def test_fit_with_insufficient_data(self):
        """Should return unfitted calibrator with insufficient data."""
        predictions = np.array([0.1, 0.2])
        actuals = np.array([0, 1])

        calibrator = IsotonicCalibrator.fit(predictions, actuals)
        assert not calibrator.is_fitted


class TestCalibrationManager:
    """Tests for CalibrationManager class."""

    def test_calibrate_with_available_group(self):
        """Should apply calibration when group is available."""
        calibrators = {
            'hour_1': IsotonicCalibrator(
                x_thresholds=[0.0, 0.1, 0.2],
                y_thresholds=[0.0, 0.18, 0.24],
                n_samples=100
            )
        }
        manager = CalibrationManager(calibrators=calibrators)

        # Hour 1 has calibration
        result = manager.calibrate(hour=1, prob=0.1)
        assert result == pytest.approx(0.18, abs=0.01)

    def test_calibrate_without_available_group(self):
        """Should clamp input when group is unavailable."""
        manager = CalibrationManager(calibrators={})

        # No calibration available - should clamp
        result = manager.calibrate(hour=1, prob=0.1)
        assert result == 0.1  # Unchanged within bounds

    def test_has_calibration(self):
        """Should correctly report calibration availability."""
        calibrators = {
            'hour_1': IsotonicCalibrator(
                x_thresholds=[0.0, 0.1],
                y_thresholds=[0.0, 0.15],
                n_samples=100
            )
        }
        manager = CalibrationManager(calibrators=calibrators)

        assert manager.has_calibration(1) is True
        assert manager.has_calibration(2) is False

    def test_save_and_load(self):
        """Should save to JSON and load back correctly."""
        calibrators = {
            'hour_1': IsotonicCalibrator(
                x_thresholds=[0.0, 0.1, 0.2],
                y_thresholds=[0.0, 0.15, 0.24],
                n_samples=100,
                brier_before=0.05,
                brier_after=0.02
            ),
            'hour_2': IsotonicCalibrator(
                x_thresholds=[0.0, 0.1],
                y_thresholds=[0.0, 0.12],
                n_samples=80
            )
        }
        manager = CalibrationManager(
            calibrators=calibrators,
            item_id=42,
            global_metrics={'brier_improvement_pct': 60.0}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'calibration.json'
            manager.save(str(path))

            # Load and verify
            loaded = CalibrationManager.load(str(path))
            assert loaded is not None
            assert loaded.item_id == 42
            assert loaded.version == '1.0'
            assert loaded.global_metrics['brier_improvement_pct'] == 60.0
            assert 'hour_1' in loaded.calibrators
            assert 'hour_2' in loaded.calibrators
            assert loaded.calibrators['hour_1'].n_samples == 100

    def test_load_nonexistent_file(self):
        """Should return None for nonexistent file."""
        result = CalibrationManager.load('/nonexistent/path/calibration.json')
        assert result is None

    def test_get_stats(self):
        """Should return summary statistics."""
        calibrators = {
            'hour_1': IsotonicCalibrator(
                x_thresholds=[0.0, 0.1],
                y_thresholds=[0.0, 0.15],
                n_samples=100,
                brier_before=0.05,
                brier_after=0.02
            ),
            'hour_2': IsotonicCalibrator(
                x_thresholds=[0.0, 0.1],
                y_thresholds=[0.0, 0.12],
                n_samples=80,
                brier_before=0.06,
                brier_after=0.03
            )
        }
        manager = CalibrationManager(calibrators=calibrators)

        stats = manager.get_stats()
        assert stats['n_groups_fitted'] == 2
        assert stats['total_samples'] == 180
        assert stats['avg_brier_before'] == pytest.approx(0.055, abs=0.001)
        assert stats['avg_brier_after'] == pytest.approx(0.025, abs=0.001)


class TestMetricFunctions:
    """Tests for metric computation functions."""

    def test_brier_score_perfect_predictions(self):
        """Brier score should be 0 for perfect predictions."""
        predictions = np.array([0.0, 1.0, 0.0, 1.0])
        actuals = np.array([0, 1, 0, 1])
        assert compute_brier_score(predictions, actuals) == 0.0

    def test_brier_score_worst_predictions(self):
        """Brier score should be 1 for completely wrong predictions."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        actuals = np.array([0, 1, 0, 1])
        assert compute_brier_score(predictions, actuals) == 1.0

    def test_brier_score_moderate_predictions(self):
        """Brier score should be between 0 and 1 for moderate predictions."""
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        actuals = np.array([0, 1, 0, 1])
        score = compute_brier_score(predictions, actuals)
        assert 0 < score < 1
        assert score == pytest.approx(0.25, abs=0.01)

    def test_calibration_error_perfect_calibration(self):
        """Calibration error should be 0 for perfectly calibrated predictions."""
        # Predictions match actual rates exactly
        predictions = np.array([0.1] * 100 + [0.9] * 100)
        actuals = np.array([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90)
        error = compute_calibration_error(predictions, actuals)
        assert error < 0.1  # Should be low

    def test_calibration_error_poor_calibration(self):
        """Calibration error should be high for poorly calibrated predictions."""
        # Always predict 0.5 but actuals are extreme
        predictions = np.array([0.5] * 100)
        actuals = np.array([0] * 10 + [1] * 90)  # 90% positive
        error = compute_calibration_error(predictions, actuals)
        assert error > 0.2  # Should be high
