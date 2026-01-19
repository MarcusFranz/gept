"""Classify prediction outcomes by comparing to actual prices."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class Prediction:
    """A prediction to evaluate."""
    prediction_time: datetime
    item_id: int
    hour_offset: int
    offset_pct: Decimal
    fill_probability: Decimal
    buy_price: Decimal
    sell_price: Decimal
    model_version: Optional[str] = None


@dataclass
class PriceWindow:
    """Actual price range during the prediction window."""
    low_price: Optional[Decimal]
    high_price: Optional[Decimal]


@dataclass
class PredictionOutcome:
    """Result of evaluating a prediction against actual prices."""
    prediction: Prediction
    price_window: PriceWindow
    outcome: str  # CLEAR_HIT, CLEAR_MISS, POSSIBLE_HIT, POSSIBLE_MISS
    buy_would_fill: bool
    sell_would_fill: bool
    evaluation_time: datetime


class OutcomeClassifier:
    """Classifies predictions based on actual price movements.

    Classification logic:
    - CLEAR_MISS: High confidence (>=70%), but price never reached target
    - CLEAR_HIT: Low confidence (<=30%), and price never reached (correctly pessimistic)
    - POSSIBLE_HIT: Price reached target (trade might have filled)
    - POSSIBLE_MISS: Very high confidence (>=90%) but price barely reached

    Note: We can't know for certain if trades filled without order book data.
    "Price reached" means the trade was POSSIBLE, not guaranteed.
    """

    def __init__(
        self,
        high_confidence_threshold: Decimal = Decimal("0.70"),
        low_confidence_threshold: Decimal = Decimal("0.30"),
        very_high_confidence_threshold: Decimal = Decimal("0.90"),
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.very_high_confidence_threshold = very_high_confidence_threshold

    def classify(
        self,
        prediction: Prediction,
        price_window: PriceWindow
    ) -> Optional[PredictionOutcome]:
        """Classify a prediction outcome based on actual prices.

        Args:
            prediction: The prediction to evaluate
            price_window: The actual price range during the prediction window

        Returns:
            PredictionOutcome with classification, or None if prices unavailable
        """
        # Can't classify without price data
        if price_window.low_price is None or price_window.high_price is None:
            return None

        # Check if prices would have allowed fills
        buy_would_fill = price_window.low_price <= prediction.buy_price
        sell_would_fill = price_window.high_price >= prediction.sell_price
        roundtrip_possible = buy_would_fill and sell_would_fill

        # Classify based on confidence vs reality
        outcome = self._determine_outcome(
            fill_probability=prediction.fill_probability,
            roundtrip_possible=roundtrip_possible,
            buy_would_fill=buy_would_fill,
            sell_would_fill=sell_would_fill,
        )

        return PredictionOutcome(
            prediction=prediction,
            price_window=price_window,
            outcome=outcome,
            buy_would_fill=buy_would_fill,
            sell_would_fill=sell_would_fill,
            evaluation_time=datetime.now(),
        )

    def _determine_outcome(
        self,
        fill_probability: Decimal,
        roundtrip_possible: bool,
        buy_would_fill: bool,
        sell_would_fill: bool,
    ) -> str:
        """Determine the outcome classification.

        Decision matrix:

        | Confidence | Roundtrip Possible | Outcome       |
        |------------|-------------------|---------------|
        | High (>=70%)| No               | CLEAR_MISS    |
        | Low (<=30%) | No               | CLEAR_HIT     |
        | Medium      | No               | CLEAR_MISS    |
        | Very High   | Yes (barely)     | POSSIBLE_MISS |
        | Any         | Yes              | POSSIBLE_HIT  |
        """
        is_high_confidence = fill_probability >= self.high_confidence_threshold
        is_low_confidence = fill_probability <= self.low_confidence_threshold
        is_very_high_confidence = fill_probability >= self.very_high_confidence_threshold

        if not roundtrip_possible:
            # Price never allowed a roundtrip
            if is_low_confidence:
                return "CLEAR_HIT"  # Correctly predicted low fill chance
            else:
                return "CLEAR_MISS"  # Predicted fill but price never reached

        # Roundtrip was possible
        if is_very_high_confidence:
            # With 90%+ confidence, we'd expect easier fills
            # Getting "barely possible" is suspiciously close
            return "POSSIBLE_MISS"

        return "POSSIBLE_HIT"
