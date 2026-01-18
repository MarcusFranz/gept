#!/usr/bin/env python3
"""
Training Window Experiment Runner
=================================

Compares model performance between different training window lengths.
Default experiment: 6 months (control) vs 1 month (treatment)

Hypothesis: For 48-hour prediction horizons, recent data (1 month) may be
more predictive than older data (6 months) due to:
- Changing game meta (updates, nerfs)
- Seasonal player behavior shifts
- Market manipulation pattern evolution

Usage:
    # Run full experiment (trains both variants on same items)
    python scripts/run_training_window_experiment.py --items 50

    # Dry run (show what would happen)
    python scripts/run_training_window_experiment.py --items 50 --dry-run

    # Custom window comparison
    python scripts/run_training_window_experiment.py --control-months 6 --treatment-months 1

    # Analyze existing experiment
    python scripts/run_training_window_experiment.py --analyze exp_20260116_123456

Requirements:
    - Database access (DB_PASS environment variable)
    - WSL training machine accessible (for GPU training)
    - Or use --local for CPU training (slower)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from db_utils import get_db_cursor, get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a training window experiment."""
    control_months: int = 6
    treatment_months: int = 1
    num_items: int = 50
    use_same_items: bool = True  # Both variants train on same items
    local_training: bool = False  # Use local CPU vs remote GPU


def create_experiment(config: ExperimentConfig) -> str:
    """Create experiment record in database, return experiment_id."""
    experiment_id = f"training_window_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with get_db_cursor() as cur:
        cur.execute("""
            INSERT INTO experiments (experiment_id, name, description, config, status)
            VALUES (%s, %s, %s, %s, 'PENDING')
            ON CONFLICT (experiment_id) DO NOTHING
            RETURNING experiment_id
        """, (
            experiment_id,
            f"Training Window: {config.control_months}mo vs {config.treatment_months}mo",
            f"Compare {config.control_months}-month training window (control) against "
            f"{config.treatment_months}-month window (treatment) on {config.num_items} items.",
            json.dumps(asdict(config))
        ))

        result = cur.fetchone()
        if result is None:
            raise RuntimeError(f"Failed to create experiment {experiment_id}")

    logger.info(f"Created experiment: {experiment_id}")
    return experiment_id


def select_experiment_items(num_items: int, months: int) -> List[int]:
    """
    Select items that qualify for training under both control and treatment windows.

    Returns items that have sufficient data for both the longer (control) and
    shorter (treatment) training windows to ensure fair comparison.
    """
    min_rows_per_month = 8640  # ~30 days * 24 hours * 12 (5-min intervals)
    min_rows = min_rows_per_month * months

    with get_db_cursor() as cur:
        # Select items with:
        # 1. Sufficient data for the specified window
        # 2. Active trading (recent volume)
        # 3. Existing 6-month model for baseline comparison
        cur.execute("""
            WITH item_stats AS (
                SELECT
                    p.item_id,
                    i.name,
                    COUNT(*) as row_count,
                    SUM(high_price_volume + low_price_volume) as total_volume
                FROM price_data_5min p
                JOIN items i ON p.item_id = i.item_id
                WHERE p.timestamp >= NOW() - INTERVAL '%s months'
                GROUP BY p.item_id, i.name
                HAVING COUNT(*) >= %s
            ),
            items_with_models AS (
                SELECT DISTINCT item_id
                FROM model_registry
                WHERE status = 'ACTIVE'
            )
            SELECT s.item_id, s.name, s.row_count, s.total_volume
            FROM item_stats s
            JOIN items_with_models m ON s.item_id = m.item_id
            ORDER BY s.total_volume DESC
            LIMIT %s
        """, (months, min_rows, num_items))

        items = [row[0] for row in cur.fetchall()]

    logger.info(f"Selected {len(items)} items for experiment")
    return items


def create_variant(
    experiment_id: str,
    variant_name: str,
    months: int,
    item_ids: List[int]
) -> str:
    """Create experiment variant and prepare training data."""
    run_id = f"exp_{variant_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with get_db_cursor() as cur:
        cur.execute("""
            INSERT INTO experiment_variants
                (experiment_id, variant_name, config, run_id, item_count)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            experiment_id,
            variant_name,
            json.dumps({'months_history': months, 'item_ids': item_ids}),
            run_id,
            len(item_ids)
        ))

    logger.info(f"Created variant '{variant_name}' with run_id: {run_id}")
    return run_id


def prepare_variant_data(
    run_id: str,
    months: int,
    item_ids: List[int],
    dry_run: bool = False
) -> bool:
    """
    Prepare training data for a variant.

    For now, this creates a temporary items file and calls the existing
    prepare_training_data.py with the --months override.
    """
    items_file = PROJECT_ROOT / 'data' / f'experiment_items_{run_id}.json'
    items_file.parent.mkdir(parents=True, exist_ok=True)

    # Write items list for the data prep script
    with open(items_file, 'w') as f:
        json.dump({'items': [{'item_id': i} for i in item_ids]}, f)

    logger.info(f"Wrote {len(item_ids)} items to {items_file}")

    if dry_run:
        logger.info(f"[DRY RUN] Would prepare data with --months {months}")
        return True

    # The actual data preparation would be done here
    # For the experiment, we'll use a simplified local approach
    logger.info(f"Preparing {months}-month data for {len(item_ids)} items...")

    # TODO: Call prepare_training_data.py with --months flag
    # For now, return True to indicate the structure is in place
    return True


def run_training(
    run_id: str,
    item_ids: List[int],
    local: bool = False,
    dry_run: bool = False
) -> bool:
    """Run training for a variant."""
    if dry_run:
        logger.info(f"[DRY RUN] Would train {len(item_ids)} items (run_id: {run_id})")
        return True

    mode = "local CPU" if local else "remote GPU"
    logger.info(f"Training {len(item_ids)} items on {mode}...")

    # TODO: Call train_runpod_multitarget.py or remote training
    # For now, return True to indicate the structure is in place
    return True


def collect_variant_results(experiment_id: str, variant_name: str, run_id: str):
    """Collect and store results for a variant."""
    with get_db_cursor() as cur:
        # Get AUC metrics from training results
        cur.execute("""
            SELECT
                COUNT(*) as item_count,
                AVG(mean_auc) as mean_auc,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mean_auc) as median_auc,
                STDDEV(mean_auc) as std_auc,
                MIN(mean_auc) as min_auc,
                MAX(mean_auc) as max_auc
            FROM model_registry
            WHERE run_id = %s
        """, (run_id,))

        row = cur.fetchone()
        if row and row[0] > 0:
            cur.execute("""
                UPDATE experiment_variants
                SET
                    item_count = %s,
                    mean_auc = %s,
                    median_auc = %s,
                    std_auc = %s,
                    min_auc = %s,
                    max_auc = %s,
                    completed_at = NOW()
                WHERE experiment_id = %s AND variant_name = %s
            """, (row[0], row[1], row[2], row[3], row[4], row[5],
                  experiment_id, variant_name))

            logger.info(f"Variant '{variant_name}' results: mean_auc={row[1]:.4f}")


def analyze_experiment(experiment_id: str) -> Dict:
    """Analyze and compare results between variants."""
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT
                variant_name,
                item_count,
                mean_auc,
                median_auc,
                std_auc,
                config
            FROM experiment_variants
            WHERE experiment_id = %s
            ORDER BY variant_name
        """, (experiment_id,))

        variants = {}
        for row in cur.fetchall():
            variants[row[0]] = {
                'item_count': row[1],
                'mean_auc': float(row[2]) if row[2] else None,
                'median_auc': float(row[3]) if row[3] else None,
                'std_auc': float(row[4]) if row[4] else None,
                'config': row[5]
            }

    if len(variants) < 2:
        logger.warning("Need at least 2 variants to compare")
        return {'variants': variants, 'comparison': None}

    # Compare control vs treatment
    control = variants.get('control_6mo', {})
    treatment = variants.get('treatment_1mo', {})

    if control.get('mean_auc') and treatment.get('mean_auc'):
        auc_diff = treatment['mean_auc'] - control['mean_auc']
        pct_change = (auc_diff / control['mean_auc']) * 100

        comparison = {
            'auc_difference': auc_diff,
            'pct_change': pct_change,
            'winner': 'treatment' if auc_diff > 0 else 'control',
            'significant': abs(auc_diff) > 0.01  # >1% difference
        }

        logger.info("=" * 60)
        logger.info("EXPERIMENT RESULTS")
        logger.info("=" * 60)
        logger.info(f"Control (6mo):   mean_auc = {control['mean_auc']:.4f}")
        logger.info(f"Treatment (1mo): mean_auc = {treatment['mean_auc']:.4f}")
        logger.info(f"Difference:      {auc_diff:+.4f} ({pct_change:+.2f}%)")
        logger.info(f"Winner:          {comparison['winner'].upper()}")
        logger.info("=" * 60)
    else:
        comparison = None

    return {'variants': variants, 'comparison': comparison}


def run_experiment(config: ExperimentConfig, dry_run: bool = False) -> str:
    """Run the full experiment."""
    logger.info("=" * 60)
    logger.info("TRAINING WINDOW EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Control:   {config.control_months} months")
    logger.info(f"Treatment: {config.treatment_months} months")
    logger.info(f"Items:     {config.num_items}")
    logger.info("=" * 60)

    # Create experiment
    experiment_id = create_experiment(config)

    # Update status to RUNNING
    with get_db_cursor() as cur:
        cur.execute("""
            UPDATE experiments
            SET status = 'RUNNING', started_at = NOW()
            WHERE experiment_id = %s
        """, (experiment_id,))

    try:
        # Select items (use the longer window to ensure both can train)
        item_ids = select_experiment_items(
            config.num_items,
            max(config.control_months, config.treatment_months)
        )

        if len(item_ids) < config.num_items:
            logger.warning(
                f"Only found {len(item_ids)} qualifying items "
                f"(requested {config.num_items})"
            )

        # Run control variant
        logger.info("\n--- CONTROL VARIANT ---")
        control_run_id = create_variant(
            experiment_id, 'control_6mo', config.control_months, item_ids
        )
        prepare_variant_data(
            control_run_id, config.control_months, item_ids, dry_run
        )
        run_training(
            control_run_id, item_ids, config.local_training, dry_run
        )

        # Run treatment variant
        logger.info("\n--- TREATMENT VARIANT ---")
        treatment_run_id = create_variant(
            experiment_id, 'treatment_1mo', config.treatment_months, item_ids
        )
        prepare_variant_data(
            treatment_run_id, config.treatment_months, item_ids, dry_run
        )
        run_training(
            treatment_run_id, item_ids, config.local_training, dry_run
        )

        # Collect results
        if not dry_run:
            collect_variant_results(experiment_id, 'control_6mo', control_run_id)
            collect_variant_results(experiment_id, 'treatment_1mo', treatment_run_id)

            # Analyze
            results = analyze_experiment(experiment_id)

            # Update experiment status
            with get_db_cursor() as cur:
                cur.execute("""
                    UPDATE experiments
                    SET status = 'COMPLETED', completed_at = NOW(), results = %s
                    WHERE experiment_id = %s
                """, (json.dumps(results), experiment_id))

        return experiment_id

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        with get_db_cursor() as cur:
            cur.execute("""
                UPDATE experiments
                SET status = 'FAILED', results = %s
                WHERE experiment_id = %s
            """, (json.dumps({'error': str(e)}), experiment_id))
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run training window experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--items', type=int, default=50,
        help='Number of items to train (default: 50)'
    )
    parser.add_argument(
        '--control-months', type=int, default=6,
        help='Training window for control variant (default: 6)'
    )
    parser.add_argument(
        '--treatment-months', type=int, default=1,
        help='Training window for treatment variant (default: 1)'
    )
    parser.add_argument(
        '--local', action='store_true',
        help='Use local CPU training instead of remote GPU'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would happen without running'
    )
    parser.add_argument(
        '--analyze', type=str, metavar='EXPERIMENT_ID',
        help='Analyze existing experiment instead of running new one'
    )

    args = parser.parse_args()

    if args.analyze:
        results = analyze_experiment(args.analyze)
        print(json.dumps(results, indent=2, default=str))
        return

    config = ExperimentConfig(
        control_months=args.control_months,
        treatment_months=args.treatment_months,
        num_items=args.items,
        local_training=args.local
    )

    experiment_id = run_experiment(config, dry_run=args.dry_run)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Analyze with: python {__file__} --analyze {experiment_id}")


if __name__ == '__main__':
    main()
