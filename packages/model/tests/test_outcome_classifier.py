"""Tests for prediction outcome classification."""

import pytest
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta

# Import will fail until we create the module
from src.validation.outcome_classifier import (
    OutcomeClassifier,
    PredictionOutcome,
    Prediction,
    PriceWindow,
)


class TestOutcomeClassification:
    """Test outcome classification logic."""

    def test_clear_miss_high_confidence_price_never_reached(self):
        """High confidence prediction, but price never came close = CLEAR_MISS."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.85"),  # High confidence
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("1100"),   # Never went low enough to buy
            high_price=Decimal("1030"),  # Never went high enough to sell
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome.outcome == "CLEAR_MISS"
        assert outcome.buy_would_fill is False
        assert outcome.sell_would_fill is False

    def test_clear_hit_low_confidence_price_never_reached(self):
        """Low confidence prediction, price never reached = CLEAR_HIT (correctly pessimistic)."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.25"),  # Low confidence
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("1100"),   # Never went low enough
            high_price=Decimal("1150"),
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome.outcome == "CLEAR_HIT"
        assert outcome.buy_would_fill is False

    def test_possible_hit_price_reached_target(self):
        """Price reached target = POSSIBLE_HIT (might have filled)."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.75"),
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("990"),   # Went below buy price
            high_price=Decimal("1050"), # Went above sell price
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome.outcome == "POSSIBLE_HIT"
        assert outcome.buy_would_fill is True
        assert outcome.sell_would_fill is True

    def test_possible_miss_very_high_confidence_price_reached(self):
        """Very high confidence but price barely reached = POSSIBLE_MISS."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.95"),  # Very high confidence
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("999"),   # Barely reached buy
            high_price=Decimal("1041"), # Barely reached sell
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        # Price reached, but with 95% confidence we'd expect easier fills
        # This is ambiguous - could be hit or miss depending on volume
        assert outcome.outcome == "POSSIBLE_MISS"

    def test_partial_fill_only_buy_reached(self):
        """Only buy price reached, not sell."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.70"),
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("990"),   # Buy would fill
            high_price=Decimal("1020"), # Sell would NOT fill
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome.buy_would_fill is True
        assert outcome.sell_would_fill is False
        # Roundtrip wouldn't complete - this is a miss
        assert outcome.outcome in ["CLEAR_MISS", "POSSIBLE_MISS"]


class TestThresholds:
    """Test classification threshold boundaries."""

    def test_high_confidence_threshold_is_70_percent(self):
        """70% is the boundary for high confidence."""
        classifier = OutcomeClassifier()
        assert classifier.high_confidence_threshold == Decimal("0.70")

    def test_low_confidence_threshold_is_30_percent(self):
        """30% is the boundary for low confidence."""
        classifier = OutcomeClassifier()
        assert classifier.low_confidence_threshold == Decimal("0.30")

    def test_exactly_70_percent_is_high_confidence(self):
        """Edge case: exactly 70% counts as high confidence."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.70"),  # Exactly at threshold
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("1100"),  # Never reached
            high_price=Decimal("1150"),
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome.outcome == "CLEAR_MISS"  # High confidence + miss


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_prices_returns_none_outcome(self):
        """If we don't have price data, can't classify."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.70"),
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=None,
            high_price=None,
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome is None

    def test_zero_fill_probability(self):
        """0% fill probability is valid (very pessimistic)."""
        prediction = Prediction(
            prediction_time=datetime.now() - timedelta(hours=4),
            item_id=2,
            hour_offset=4,
            offset_pct=Decimal("0.02"),
            fill_probability=Decimal("0.00"),
            buy_price=Decimal("1000"),
            sell_price=Decimal("1040"),
        )

        price_window = PriceWindow(
            low_price=Decimal("1100"),
            high_price=Decimal("1150"),
        )

        classifier = OutcomeClassifier()
        outcome = classifier.classify(prediction, price_window)

        assert outcome.outcome == "CLEAR_HIT"  # Correctly predicted no fill
