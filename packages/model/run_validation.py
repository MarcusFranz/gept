#!/usr/bin/env python3
"""Run the prediction validation job.

This script evaluates past predictions against actual price data
and saves the outcomes for analysis.

Usage:
    python run_validation.py [--dry-run] [--hours 4,8,24]

Options:
    --dry-run       Print what would be done without saving to DB
    --hours         Comma-separated list of lookback hours (default: 4,8,12,24,48)
    --db-url        Database URL (default: from DATABASE_URL env var)
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from src.validation import (
    ValidationJob,
    ValidationConfig,
    ValidationDBAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run prediction validation job'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without saving'
    )
    parser.add_argument(
        '--hours',
        type=str,
        default='4,8,12,24,48',
        help='Comma-separated lookback hours (default: 4,8,12,24,48)'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.environ.get('DATABASE_URL'),
        help='Database URL (default: DATABASE_URL env var)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if not args.db_url:
        logger.error("No database URL provided. Set DATABASE_URL or use --db-url")
        sys.exit(1)

    # Parse lookback hours
    lookback_hours = [int(h.strip()) for h in args.hours.split(',')]
    logger.info(f"Validation job starting at {datetime.now()}")
    logger.info(f"Lookback hours: {lookback_hours}")
    logger.info(f"Dry run: {args.dry_run}")

    # Connect to database
    try:
        conn = psycopg2.connect(args.db_url, cursor_factory=RealDictCursor)
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Create job
        config = ValidationConfig(lookback_hours=lookback_hours)
        db_adapter = ValidationDBAdapter(connection=conn)
        job = ValidationJob(db=db_adapter, config=config)

        if args.dry_run:
            # Just fetch and classify, don't save
            logger.info("DRY RUN - fetching predictions to evaluate...")
            predictions = job.fetch_predictions_to_evaluate()
            logger.info(f"Would evaluate {len(predictions)} predictions")
            for p in predictions[:10]:  # Show first 10
                logger.info(f"  Item {p.item_id}: {p.hour_offset}h, {float(p.fill_probability):.2%} confidence")
            if len(predictions) > 10:
                logger.info(f"  ... and {len(predictions) - 10} more")
        else:
            # Run full job
            result = job.run()

            logger.info("=" * 50)
            logger.info("VALIDATION JOB COMPLETE")
            logger.info("=" * 50)
            logger.info(f"Duration: {result.duration_seconds:.2f}s")
            logger.info(f"Predictions evaluated: {result.total_evaluated}")
            logger.info(f"Outcomes saved: {result.total_saved}")
            logger.info(f"Duplicates skipped: {result.total_skipped}")
            logger.info("Outcomes by type:")
            for outcome_type, count in result.outcomes_by_type.items():
                pct = count / max(result.total_evaluated, 1) * 100
                logger.info(f"  {outcome_type}: {count} ({pct:.1f}%)")

            if result.errors:
                logger.warning(f"Errors: {len(result.errors)}")
                for error in result.errors:
                    logger.warning(f"  {error}")

    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()
