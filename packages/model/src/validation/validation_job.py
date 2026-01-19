"""Validation job that evaluates predictions against actual prices."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import time

from .outcome_classifier import (
    OutcomeClassifier,
    Prediction,
    PriceWindow,
    PredictionOutcome,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for the validation job."""

    # Which prediction windows to evaluate (hours ago)
    lookback_hours: List[int] = field(
        default_factory=lambda: [4, 8, 12, 24, 48]
    )

    # Batch size for DB operations
    batch_size: int = 1000

    # Classifier thresholds
    high_confidence_threshold: Decimal = Decimal("0.70")
    low_confidence_threshold: Decimal = Decimal("0.30")


@dataclass
class ValidationResult:
    """Result of a validation job run."""

    lookback_hours_processed: List[int]
    total_evaluated: int
    total_saved: int
    total_skipped: int
    outcomes_by_type: Dict[str, int]
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


class ValidationJob:
    """Job that evaluates past predictions against actual prices.

    Run hourly to evaluate predictions that have now completed their
    time windows. For example, a 4-hour prediction made 4 hours ago
    can now be evaluated.
    """

    def __init__(self, db, config: Optional[ValidationConfig] = None):
        """Initialize the validation job.

        Args:
            db: Database connection with methods:
                - fetch_predictions_for_evaluation(hours_ago, limit)
                - fetch_price_range(item_id, start_time, end_time)
                - save_prediction_outcomes(outcomes)
            config: Job configuration
        """
        self.db = db
        self.config = config or ValidationConfig()
        self.classifier = OutcomeClassifier(
            high_confidence_threshold=self.config.high_confidence_threshold,
            low_confidence_threshold=self.config.low_confidence_threshold,
        )

    def run(self) -> ValidationResult:
        """Run the validation job.

        Returns:
            ValidationResult with summary statistics
        """
        start_time = time.time()

        all_outcomes: List[PredictionOutcome] = []
        errors: List[str] = []

        for hours in self.config.lookback_hours:
            try:
                outcomes = self._evaluate_predictions_for_hours(hours)
                all_outcomes.extend(outcomes)
                logger.info(
                    f"Evaluated {len(outcomes)} predictions for {hours}h window"
                )
            except Exception as e:
                error_msg = f"Error evaluating {hours}h predictions: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Save all outcomes
        saved, skipped = self.save_outcomes(all_outcomes)

        # Compute summary
        outcomes_by_type = self._count_outcomes_by_type(all_outcomes)

        return ValidationResult(
            lookback_hours_processed=self.config.lookback_hours,
            total_evaluated=len(all_outcomes),
            total_saved=saved,
            total_skipped=skipped,
            outcomes_by_type=outcomes_by_type,
            duration_seconds=time.time() - start_time,
            errors=errors,
        )

    def _evaluate_predictions_for_hours(
        self, hours_ago: int
    ) -> List[PredictionOutcome]:
        """Evaluate predictions made hours_ago that predicted hours_ago window.

        Example: If hours_ago=4, we look at predictions made 4 hours ago
        that had hour_offset=4 (predicting 4 hours into the future).
        """
        predictions = self.fetch_predictions_to_evaluate(hours_ago)
        outcomes = []

        for prediction in predictions:
            try:
                price_window = self.fetch_price_window(prediction)
                outcome = self.classifier.classify(prediction, price_window)
                if outcome is not None:
                    outcomes.append(outcome)
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate prediction for item {prediction.item_id}: {e}"
                )

        return outcomes

    def fetch_predictions_to_evaluate(
        self, hours_ago: Optional[int] = None
    ) -> List[Prediction]:
        """Fetch predictions that need evaluation.

        If hours_ago is specified, fetches predictions made that many hours ago
        with matching hour_offset. Otherwise, fetches for all configured hours.
        """
        if hours_ago is not None:
            return self.db.fetch_predictions_for_evaluation(
                hours_ago=hours_ago,
                limit=self.config.batch_size * 10,  # Fetch more than batch
            )

        # Fetch for all lookback hours
        all_predictions = []
        for hours in self.config.lookback_hours:
            predictions = self.db.fetch_predictions_for_evaluation(
                hours_ago=hours,
                limit=self.config.batch_size * 10,
            )
            all_predictions.extend(predictions)

        return all_predictions

    def fetch_price_window(self, prediction: Prediction) -> PriceWindow:
        """Fetch actual price range for the prediction window."""
        start_time = prediction.prediction_time
        end_time = start_time + timedelta(hours=prediction.hour_offset)

        return self.db.fetch_price_range(
            item_id=prediction.item_id,
            start_time=start_time,
            end_time=end_time,
        )

    def save_outcomes(
        self, outcomes: List[PredictionOutcome]
    ) -> Tuple[int, int]:
        """Save outcomes to database in batches.

        Returns:
            Tuple of (saved_count, skipped_count)
        """
        if not outcomes:
            return 0, 0

        saved = 0
        skipped = 0

        for i in range(0, len(outcomes), self.config.batch_size):
            batch = outcomes[i:i + self.config.batch_size]
            try:
                self.db.save_prediction_outcomes(batch)
                saved += len(batch)
            except Exception as e:
                if "duplicate" in str(e).lower():
                    logger.info(f"Skipping {len(batch)} duplicate outcomes")
                    skipped += len(batch)
                else:
                    raise

        return saved, skipped

    def _count_outcomes_by_type(
        self, outcomes: List[PredictionOutcome]
    ) -> Dict[str, int]:
        """Count outcomes by classification type."""
        counts: Dict[str, int] = {
            "CLEAR_HIT": 0,
            "CLEAR_MISS": 0,
            "POSSIBLE_HIT": 0,
            "POSSIBLE_MISS": 0,
        }

        for outcome in outcomes:
            if outcome.outcome in counts:
                counts[outcome.outcome] += 1

        return counts
