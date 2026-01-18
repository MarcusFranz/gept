#!/usr/bin/env python3
"""
Prediction Evaluation Job

Evaluates past predictions against actual price data.
Runs hourly to check if predictions from 24-48 hours ago would have filled.

This provides:
1. Calibration monitoring - Are predicted probabilities matching actual fill rates?
2. Performance tracking - How many predictions filled?
3. Drift detection - Is model performance degrading over time?

Usage:
    python evaluation_job.py                    # Evaluate last 24h of predictions
    python evaluation_job.py --hours 48         # Evaluate last 48h
    python evaluation_job.py --dry-run          # Show results without saving

Cron setup:
    0 * * * * cd /path/to/GePT\ Model && python3 src/evaluation_job.py >> logs/evaluation.log 2>&1
"""

import sys
import time
import argparse
import logging
from datetime import datetime
from typing import List

from psycopg2.extras import execute_values
import pandas as pd

# Centralized database connection management
from db_utils import get_simple_connection

# Calibration bucket boundaries
CALIBRATION_BUCKETS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 1.0]


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class PredictionEvaluator:
    """Evaluates past predictions against actual price data."""

    def __init__(self):
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Open database connection (non-pooled for long-running job)."""
        self.conn = get_simple_connection()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def fetch_predictions_to_evaluate(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Fetch predictions from 24-48 hours ago that haven't been evaluated yet.

        We evaluate predictions after their target_hour has passed so we can
        check against actual price data.
        """
        query = """
            SELECT
                p.time as prediction_time,
                p.item_id,
                p.hour_offset,
                p.offset_pct,
                p.fill_probability,
                p.buy_price as buy_target,
                p.sell_price as sell_target,
                p.target_hour
            FROM predictions p
            WHERE p.target_hour < NOW() - INTERVAL '1 hour'
              AND p.time > NOW() - make_interval(hours => %s)
              AND NOT EXISTS (
                  SELECT 1 FROM actual_fills af
                  WHERE af.prediction_time = p.time
                    AND af.item_id = p.item_id
                    AND af.hour_offset = p.hour_offset
                    AND af.offset_pct = p.offset_pct
              )
            ORDER BY p.time
            LIMIT 10000
        """

        df = pd.read_sql(query, self.conn, params=[hours_back + 48])
        return df

    def fetch_actual_prices(self, item_id: int, start_time: datetime,
                            end_time: datetime) -> pd.DataFrame:
        """Fetch actual price data for evaluation window."""
        query = """
            SELECT timestamp, avg_high_price as high, avg_low_price as low
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp
        """

        df = pd.read_sql(query, self.conn, params=[item_id, start_time, end_time])
        return df

    def evaluate_prediction(self, pred: dict) -> dict:
        """
        Evaluate a single prediction against actual price data.

        Args:
            pred: Prediction dict with prediction_time, item_id, hour_offset,
                  offset_pct, fill_probability, buy_target, sell_target, target_hour

        Returns:
            Evaluation result dict
        """
        # Get actual prices for the prediction window
        # Window: from prediction_time to prediction_time + hour_offset
        start_time = pred['prediction_time']
        end_time = pred['target_hour']

        prices = self.fetch_actual_prices(pred['item_id'], start_time, end_time)

        if len(prices) == 0:
            return None

        # Calculate actual min/max in window
        actual_min_low = prices['low'].min()
        actual_max_high = prices['high'].max()

        # Would orders have filled?
        buy_would_fill = actual_min_low <= pred['buy_target']
        sell_would_fill = actual_max_high >= pred['sell_target']
        both_would_fill = buy_would_fill and sell_would_fill

        return {
            'time': datetime.now(),
            'prediction_time': pred['prediction_time'],
            'item_id': pred['item_id'],
            'hour_offset': pred['hour_offset'],
            'offset_pct': float(pred['offset_pct']),
            'predicted_probability': float(pred['fill_probability']),
            'buy_target': float(pred['buy_target']),
            'sell_target': float(pred['sell_target']),
            'actual_min_low': float(actual_min_low) if not pd.isna(actual_min_low) else None,
            'actual_max_high': float(actual_max_high) if not pd.isna(actual_max_high) else None,
            'buy_would_fill': buy_would_fill,
            'sell_would_fill': sell_would_fill,
            'both_would_fill': both_would_fill
        }

    def save_evaluations(self, evaluations: List[dict]):
        """Save evaluation results to database."""
        if not evaluations:
            return

        values = [
            (e['time'], e['prediction_time'], e['item_id'], e['hour_offset'],
             e['offset_pct'], e['predicted_probability'], e['buy_target'],
             e['sell_target'], e['actual_min_low'], e['actual_max_high'],
             e['buy_would_fill'], e['sell_would_fill'], e['both_would_fill'])
            for e in evaluations
        ]

        cur = self.conn.cursor()
        try:
            execute_values(cur, """
                INSERT INTO actual_fills (
                    time, prediction_time, item_id, hour_offset, offset_pct,
                    predicted_probability, buy_target, sell_target,
                    actual_min_low, actual_max_high,
                    buy_would_fill, sell_would_fill, both_would_fill
                ) VALUES %s
                ON CONFLICT DO NOTHING
            """, values)
            self.conn.commit()
        finally:
            cur.close()

        self.logger.info(f"Saved {len(evaluations)} evaluations to database")

    def compute_calibration_metrics(self, evaluations: List[dict]) -> List[dict]:
        """Compute calibration metrics by probability bucket."""
        if not evaluations:
            return []

        df = pd.DataFrame(evaluations)
        metrics = []

        for i in range(len(CALIBRATION_BUCKETS) - 1):
            low, high = CALIBRATION_BUCKETS[i], CALIBRATION_BUCKETS[i+1]

            mask = (df['predicted_probability'] >= low) & (df['predicted_probability'] < high)
            bucket_df = df[mask]

            if len(bucket_df) > 0:
                avg_predicted = bucket_df['predicted_probability'].mean()
                actual_rate = bucket_df['both_would_fill'].mean()
                calibration_error = actual_rate - avg_predicted

                metrics.append({
                    'time': datetime.now(),
                    'bucket_low': low,
                    'bucket_high': high,
                    'prediction_count': len(bucket_df),
                    'avg_predicted_prob': avg_predicted,
                    'actual_fill_rate': actual_rate,
                    'calibration_error': calibration_error
                })

        return metrics

    def save_calibration_metrics(self, metrics: List[dict]):
        """Save calibration metrics to database."""
        if not metrics:
            return

        values = [
            (m['time'], m['bucket_low'], m['bucket_high'], m['prediction_count'],
             m['avg_predicted_prob'], m['actual_fill_rate'], m['calibration_error'])
            for m in metrics
        ]

        cur = self.conn.cursor()
        try:
            execute_values(cur, """
                INSERT INTO calibration_metrics (
                    time, bucket_low, bucket_high, prediction_count,
                    avg_predicted_prob, actual_fill_rate, calibration_error
                ) VALUES %s
                ON CONFLICT DO NOTHING
            """, values)
            self.conn.commit()
        finally:
            cur.close()

    def check_calibration_drift(self, metrics: List[dict]) -> List[str]:
        """Check for calibration drift and return alerts."""
        alerts = []

        for m in metrics:
            # Alert if calibration error is large (>25% relative)
            if m['prediction_count'] >= 50:  # Need enough samples
                if abs(m['calibration_error']) > 0.10:  # >10% absolute error
                    rel_error = abs(m['calibration_error']) / max(m['avg_predicted_prob'], 0.01)
                    if rel_error > 0.25:
                        bucket_name = f"{m['bucket_low']:.0%}-{m['bucket_high']:.0%}"
                        alerts.append(
                            f"Calibration drift in {bucket_name}: "
                            f"predicted {m['avg_predicted_prob']:.1%}, "
                            f"actual {m['actual_fill_rate']:.1%} "
                            f"(error: {m['calibration_error']:+.1%})"
                        )

        return alerts


def main():
    parser = argparse.ArgumentParser(description='Evaluate past predictions')
    parser.add_argument('--hours', type=int, default=24,
                        help='Hours of predictions to evaluate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show results without saving to database')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    start_time = time.time()

    logger.info("="*60)
    logger.info("PREDICTION EVALUATION JOB")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("="*60)

    evaluator = PredictionEvaluator()

    try:
        evaluator.connect()

        # Fetch predictions to evaluate
        logger.info(f"Fetching predictions from last {args.hours}h...")
        predictions_df = evaluator.fetch_predictions_to_evaluate(args.hours)
        logger.info(f"Found {len(predictions_df)} predictions to evaluate")

        if len(predictions_df) == 0:
            logger.info("No predictions to evaluate")
            return

        # Evaluate each prediction
        logger.info("Evaluating predictions...")
        evaluations = []
        items_processed = set()

        for idx, row in predictions_df.iterrows():
            pred = row.to_dict()

            try:
                result = evaluator.evaluate_prediction(pred)
                if result:
                    evaluations.append(result)
                    items_processed.add(pred['item_id'])
            except Exception as e:
                logger.debug(f"Error evaluating prediction: {e}")
                continue

            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1}/{len(predictions_df)} predictions...")

        logger.info(f"Evaluated {len(evaluations)} predictions across {len(items_processed)} items")

        # Compute calibration metrics
        calibration = evaluator.compute_calibration_metrics(evaluations)

        # Print summary
        logger.info("\nCalibration Summary:")
        logger.info(f"{'Bucket':<12} {'Count':>8} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
        logger.info("-" * 54)

        for m in calibration:
            bucket = f"{m['bucket_low']:.0%}-{m['bucket_high']:.0%}"
            logger.info(
                f"{bucket:<12} {m['prediction_count']:>8} "
                f"{m['avg_predicted_prob']:>10.2%} "
                f"{m['actual_fill_rate']:>10.2%} "
                f"{m['calibration_error']:>+10.2%}"
            )

        # Overall stats
        if evaluations:
            df = pd.DataFrame(evaluations)
            total_fills = df['both_would_fill'].sum()
            fill_rate = df['both_would_fill'].mean()
            avg_predicted = df['predicted_probability'].mean()

            logger.info("\nOverall:")
            logger.info(f"  Total predictions: {len(evaluations)}")
            logger.info(f"  Total fills: {total_fills}")
            logger.info(f"  Actual fill rate: {fill_rate:.2%}")
            logger.info(f"  Avg predicted: {avg_predicted:.2%}")
            logger.info(f"  Calibration error: {fill_rate - avg_predicted:+.2%}")

        # Check for drift
        alerts = evaluator.check_calibration_drift(calibration)
        if alerts:
            logger.warning("\n⚠️  CALIBRATION ALERTS:")
            for alert in alerts:
                logger.warning(f"  {alert}")

        # Save results
        if args.dry_run:
            logger.info("\nDRY RUN - Results not saved")
        else:
            logger.info("\nSaving results to database...")
            evaluator.save_evaluations(evaluations)
            evaluator.save_calibration_metrics(calibration)
            logger.info("Done")

    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        sys.exit(1)
    finally:
        evaluator.close()

    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed:.1f}s")
    logger.info("="*60)


if __name__ == "__main__":
    main()
