"""Tests for validation database adapter."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from src.validation.db_adapter import ValidationDBAdapter
from src.validation.outcome_classifier import (
    Prediction,
    PriceWindow,
    PredictionOutcome,
)


class TestFetchPredictionsForEvaluation:
    """Test fetching predictions that need evaluation."""

    def test_builds_correct_query(self):
        """Should query for predictions made hours_ago with matching hour_offset."""
        mock_conn = Mock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_cursor.fetchall.return_value = []

        adapter = ValidationDBAdapter(connection=mock_conn)
        adapter.fetch_predictions_for_evaluation(hours_ago=4, limit=100)

        # Verify the query includes the time filter
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]

        assert "hour_offset" in query
        assert "time" in query

    def test_returns_prediction_objects(self):
        """Should return list of Prediction dataclass objects."""
        mock_conn = Mock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        # Mock a row from the DB
        mock_cursor.fetchall.return_value = [
            {
                'time': datetime.now() - timedelta(hours=4),
                'item_id': 2,
                'hour_offset': 4,
                'offset_pct': Decimal("0.02"),
                'fill_probability': Decimal("0.75"),
                'buy_price': Decimal("1000"),
                'sell_price': Decimal("1040"),
                'model_version': 'v1.0',
            }
        ]
        mock_cursor.description = [
            ('time',), ('item_id',), ('hour_offset',), ('offset_pct',),
            ('fill_probability',), ('buy_price',), ('sell_price',), ('model_version',),
        ]

        adapter = ValidationDBAdapter(connection=mock_conn)
        predictions = adapter.fetch_predictions_for_evaluation(hours_ago=4, limit=100)

        assert len(predictions) == 1
        assert isinstance(predictions[0], Prediction)
        assert predictions[0].item_id == 2


class TestFetchPriceRange:
    """Test fetching actual price range for evaluation."""

    def test_returns_price_window(self):
        """Should return PriceWindow with min/max prices."""
        mock_conn = Mock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_cursor.fetchone.return_value = {
            'min_price': Decimal("990"),
            'max_price': Decimal("1050"),
        }

        adapter = ValidationDBAdapter(connection=mock_conn)
        price_window = adapter.fetch_price_range(
            item_id=2,
            start_time=datetime.now() - timedelta(hours=4),
            end_time=datetime.now(),
        )

        assert isinstance(price_window, PriceWindow)
        assert price_window.low_price == Decimal("990")
        assert price_window.high_price == Decimal("1050")

    def test_handles_no_price_data(self):
        """Should return PriceWindow with None values if no data."""
        mock_conn = Mock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_cursor.fetchone.return_value = None

        adapter = ValidationDBAdapter(connection=mock_conn)
        price_window = adapter.fetch_price_range(
            item_id=99999,  # Non-existent item
            start_time=datetime.now() - timedelta(hours=4),
            end_time=datetime.now(),
        )

        assert price_window.low_price is None
        assert price_window.high_price is None


class TestSavePredictionOutcomes:
    """Test saving outcomes to database."""

    def test_inserts_outcomes(self):
        """Should INSERT outcomes into prediction_outcomes table."""
        mock_conn = Mock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        outcome = PredictionOutcome(
            prediction=Prediction(
                prediction_time=datetime.now() - timedelta(hours=4),
                item_id=2,
                hour_offset=4,
                offset_pct=Decimal("0.02"),
                fill_probability=Decimal("0.75"),
                buy_price=Decimal("1000"),
                sell_price=Decimal("1040"),
            ),
            price_window=PriceWindow(
                low_price=Decimal("990"),
                high_price=Decimal("1050"),
            ),
            outcome="POSSIBLE_HIT",
            buy_would_fill=True,
            sell_would_fill=True,
            evaluation_time=datetime.now(),
        )

        adapter = ValidationDBAdapter(connection=mock_conn)
        adapter.save_prediction_outcomes([outcome])

        call_args = mock_cursor.executemany.call_args
        query = call_args[0][0]

        assert "INSERT INTO prediction_outcomes" in query
        assert "ON CONFLICT" in query  # Upsert for idempotency
