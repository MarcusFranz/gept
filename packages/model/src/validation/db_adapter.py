"""Database adapter for validation operations."""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Any

from .outcome_classifier import Prediction, PriceWindow, PredictionOutcome

logger = logging.getLogger(__name__)


class ValidationDBAdapter:
    """Database adapter for validation job operations.

    Provides methods to:
    - Fetch predictions that need evaluation
    - Fetch actual price ranges for prediction windows
    - Save prediction outcomes
    """

    def __init__(self, connection):
        """Initialize with a database connection.

        Args:
            connection: A database connection (psycopg2 or similar)
        """
        self.connection = connection

    def fetch_predictions_for_evaluation(
        self,
        hours_ago: int,
        limit: int = 10000,
    ) -> List[Prediction]:
        """Fetch predictions made hours_ago that predicted hours_ago window.

        We evaluate predictions where:
        - prediction was made ~hours_ago
        - the prediction's hour_offset matches (so window has completed)
        - not already evaluated (not in prediction_outcomes)

        Args:
            hours_ago: How many hours ago the prediction was made
            limit: Maximum predictions to fetch

        Returns:
            List of Prediction objects to evaluate
        """
        # Allow 10 minute window for timing flexibility
        time_start = datetime.now() - timedelta(hours=hours_ago, minutes=10)
        time_end = datetime.now() - timedelta(hours=hours_ago, minutes=-10)

        query = """
            SELECT
                p.time,
                p.item_id,
                p.hour_offset,
                p.offset_pct,
                p.fill_probability,
                p.buy_price,
                p.sell_price,
                p.model_version
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON (
                po.prediction_time = p.time
                AND po.item_id = p.item_id
                AND po.hour_offset = p.hour_offset
                AND po.offset_pct = p.offset_pct
            )
            WHERE p.time BETWEEN %s AND %s
              AND p.hour_offset = %s
              AND po.id IS NULL  -- Not already evaluated
            ORDER BY p.time DESC
            LIMIT %s
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, (time_start, time_end, hours_ago, limit))
            rows = cursor.fetchall()

            # Handle both dict and tuple cursors
            if rows and isinstance(rows[0], dict):
                return [self._row_to_prediction(row) for row in rows]
            elif rows:
                columns = [desc[0] for desc in cursor.description]
                return [
                    self._row_to_prediction(dict(zip(columns, row)))
                    for row in rows
                ]
            return []

    def fetch_price_range(
        self,
        item_id: int,
        start_time: datetime,
        end_time: datetime,
    ) -> PriceWindow:
        """Fetch the actual price range during a time window.

        Uses price_data_5min to get min low and max high prices.

        Args:
            item_id: The item to check
            start_time: Window start
            end_time: Window end

        Returns:
            PriceWindow with low/high prices (None if no data)
        """
        query = """
            SELECT
                MIN(avg_low_price) as min_price,
                MAX(avg_high_price) as max_price
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp BETWEEN %s AND %s
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, (item_id, start_time, end_time))
            row = cursor.fetchone()

            if row is None:
                return PriceWindow(low_price=None, high_price=None)

            # Handle dict or tuple
            if isinstance(row, dict):
                return PriceWindow(
                    low_price=row.get('min_price'),
                    high_price=row.get('max_price'),
                )
            else:
                return PriceWindow(
                    low_price=row[0],
                    high_price=row[1],
                )

    def save_prediction_outcomes(
        self,
        outcomes: List[PredictionOutcome],
    ) -> None:
        """Save prediction outcomes to database.

        Uses ON CONFLICT DO NOTHING for idempotency.

        Args:
            outcomes: List of outcomes to save
        """
        if not outcomes:
            return

        query = """
            INSERT INTO prediction_outcomes (
                prediction_time,
                item_id,
                hour_offset,
                offset_pct,
                predicted_fill_probability,
                predicted_buy_price,
                predicted_sell_price,
                evaluation_time,
                actual_low_price,
                actual_high_price,
                buy_would_fill,
                sell_would_fill,
                outcome,
                model_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (prediction_time, item_id, hour_offset, offset_pct)
            DO NOTHING
        """

        values = [
            (
                o.prediction.prediction_time,
                o.prediction.item_id,
                o.prediction.hour_offset,
                o.prediction.offset_pct,
                o.prediction.fill_probability,
                o.prediction.buy_price,
                o.prediction.sell_price,
                o.evaluation_time,
                o.price_window.low_price,
                o.price_window.high_price,
                o.buy_would_fill,
                o.sell_would_fill,
                o.outcome,
                o.prediction.model_version,
            )
            for o in outcomes
        ]

        with self.connection.cursor() as cursor:
            cursor.executemany(query, values)

        self.connection.commit()

    def _row_to_prediction(self, row: dict) -> Prediction:
        """Convert a database row to a Prediction object."""
        return Prediction(
            prediction_time=row['time'],
            item_id=row['item_id'],
            hour_offset=row['hour_offset'],
            offset_pct=Decimal(str(row['offset_pct'])),
            fill_probability=Decimal(str(row['fill_probability'])),
            buy_price=Decimal(str(row['buy_price'])),
            sell_price=Decimal(str(row['sell_price'])),
            model_version=row.get('model_version'),
        )
