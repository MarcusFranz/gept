"""Tests for the validation job that evaluates past predictions."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from src.validation.validation_job import (
    ValidationJob,
    ValidationConfig,
)
from src.validation.outcome_classifier import Prediction, PriceWindow


class TestValidationJobConfig:
    """Test validation job configuration."""

    def test_default_lookback_hours(self):
        """Default lookback includes common prediction windows."""
        config = ValidationConfig()
        assert 4 in config.lookback_hours
        assert 8 in config.lookback_hours
        assert 24 in config.lookback_hours
        assert 48 in config.lookback_hours

    def test_batch_size_default(self):
        """Default batch size for DB operations."""
        config = ValidationConfig()
        assert config.batch_size == 1000


class TestValidationJobFetchPredictions:
    """Test fetching predictions to evaluate."""

    def test_fetches_predictions_for_lookback_hours(self):
        """Should fetch predictions made N hours ago for each N in lookback_hours."""
        mock_db = Mock()
        mock_db.fetch_predictions_for_evaluation.return_value = []

        config = ValidationConfig(lookback_hours=[4, 8])
        job = ValidationJob(db=mock_db, config=config)

        job.fetch_predictions_to_evaluate()

        # Should have called for both 4h and 8h lookback
        assert mock_db.fetch_predictions_for_evaluation.call_count == 2


class TestValidationJobFetchPrices:
    """Test fetching actual prices for evaluation."""

    def test_fetches_price_range_for_window(self):
        """Should fetch min/max prices for the prediction window."""
        mock_db = Mock()
        mock_db.fetch_price_range.return_value = PriceWindow(
            low_price=Decimal("1000"),
            high_price=Decimal("1100"),
        )

        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.75"),
            buy_price=Decimal("1050"),
            sell_price=Decimal("1100"),
        )

        config = ValidationConfig()
        job = ValidationJob(db=mock_db, config=config)

        price_window = job.fetch_price_window(prediction)

        mock_db.fetch_price_range.assert_called_once()
        assert price_window.low_price == Decimal("1000")


class TestValidationJobSaveOutcomes:
    """Test saving outcomes to database."""

    def test_saves_outcomes_in_batches(self):
        """Should save outcomes to DB in configurable batch sizes."""
        mock_db = Mock()

        config = ValidationConfig(batch_size=2)
        job = ValidationJob(db=mock_db, config=config)

        # Create 5 mock outcomes
        outcomes = [Mock() for _ in range(5)]

        job.save_outcomes(outcomes)

        # Should save in 3 batches (2, 2, 1)
        assert mock_db.save_prediction_outcomes.call_count == 3

    def test_skips_duplicates(self):
        """Should handle duplicate key errors gracefully."""
        mock_db = Mock()
        mock_db.save_prediction_outcomes.side_effect = [
            None,  # First batch succeeds
            Exception("duplicate key"),  # Second batch has duplicates
            None,  # Third batch succeeds
        ]

        config = ValidationConfig(batch_size=2)
        job = ValidationJob(db=mock_db, config=config)

        outcomes = [Mock() for _ in range(5)]

        # Should not raise, should log and continue
        saved, skipped = job.save_outcomes(outcomes)

        assert saved == 3  # First batch (2) + third batch (1)
        assert skipped == 2  # Second batch


class TestValidationJobRun:
    """Test full validation job run."""

    def test_run_processes_all_lookback_hours(self):
        """Full run should process predictions for all configured hours."""
        mock_db = Mock()
        mock_db.fetch_predictions_for_evaluation.return_value = []
        mock_db.save_prediction_outcomes.return_value = None

        config = ValidationConfig(lookback_hours=[4, 8, 24])
        job = ValidationJob(db=mock_db, config=config)

        result = job.run()

        assert result.lookback_hours_processed == [4, 8, 24]

    def test_run_returns_summary(self):
        """Run should return summary of outcomes."""
        mock_db = Mock()
        mock_db.fetch_predictions_for_evaluation.return_value = []

        config = ValidationConfig(lookback_hours=[4])
        job = ValidationJob(db=mock_db, config=config)

        result = job.run()

        assert hasattr(result, 'total_evaluated')
        assert hasattr(result, 'outcomes_by_type')
        assert hasattr(result, 'duration_seconds')
