#!/usr/bin/env python3
"""
Hyperparameter Sweep for GePT Models
=====================================

Comprehensive sweep to find optimal model configurations across:
- Training window lengths (7, 14, 30, 60, 90, 180 days)
- CatBoost hyperparameters (iterations, depth, learning_rate)
- Item categories (high-volume, high-value, consumables, equipment)

Usage:
    # Full sweep (all combinations - takes many hours)
    python scripts/run_hyperparameter_sweep.py --full

    # Quick sweep (key configurations only)
    python scripts/run_hyperparameter_sweep.py --quick

    # Training window sweep only
    python scripts/run_hyperparameter_sweep.py --windows-only --items 50

    # Test specific configuration
    python scripts/run_hyperparameter_sweep.py --days 30 --iterations 200 --depth 6

    # Dry run
    python scripts/run_hyperparameter_sweep.py --quick --dry-run

Requirements:
    - Database access (DB_PASS environment variable)
    - WSL training machine with GPU
    - Run migrations/012_experiments.sql first
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from itertools import product
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


# =============================================================================
# Item Categories
# =============================================================================

# High-volume items (frequently traded, liquid market)
HIGH_VOLUME_ITEMS = [
    561,    # Nature rune
    560,    # Death rune
    562,    # Chaos rune
    2,      # Cannonball
    1515,   # Yew logs
    1513,   # Magic logs
    386,    # Shark
    3024,   # Super restore(4)
    2434,   # Prayer potion(4)
    12695,  # Ranarr seed
]

# High-value items (expensive, lower volume)
HIGH_VALUE_ITEMS = [
    22325,  # Scythe of vitur
    20997,  # Twisted bow
    12825,  # Arcane spirit shield
    11802,  # Armadyl godsword
    11804,  # Bandos godsword
    12002,  # Occult necklace
    21034,  # Ancestral robe top
    22486,  # Torva full helm
    21006,  # Sanguinesti staff
    22324,  # Ghrazi rapier
]

# Consumables (food, potions)
CONSUMABLE_ITEMS = [
    385,    # Shark
    379,    # Lobster
    391,    # Manta ray
    3024,   # Super restore(4)
    2434,   # Prayer potion(4)
    12695,  # Ranarr seed
    5295,   # Ranarr weed
    2481,   # Zamorak brew(4)
    12911,  # Stamina potion(4)
    3041,   # Ranging potion(4)
]

# Equipment (armor, weapons)
EQUIPMENT_ITEMS = [
    11832,  # Bandos chestplate
    11834,  # Bandos tassets
    11828,  # Armadyl helmet
    11830,  # Armadyl chestplate
    10330,  # Pegasian boots
    12006,  # Abyssal dagger
    22296,  # Avernic defender
    21018,  # Ancestral hat
    11785,  # Armadyl crossbow
    4151,   # Abyssal whip
]


# =============================================================================
# Sweep Configurations
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration for a hyperparameter sweep."""
    name: str
    description: str

    # Training windows to test (in days)
    training_days: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90, 180])

    # CatBoost core hyperparameters
    iterations_list: List[int] = field(default_factory=lambda: [200])
    depth_list: List[int] = field(default_factory=lambda: [6])
    learning_rate_list: List[float] = field(default_factory=lambda: [0.1])

    # CatBoost regularization
    l2_leaf_reg_list: List[float] = field(default_factory=lambda: [3.0])  # L2 regularization (1, 3, 10)
    min_data_in_leaf_list: List[int] = field(default_factory=lambda: [1])  # Min samples per leaf (1, 5, 10)

    # CatBoost sampling/bagging
    subsample_list: List[float] = field(default_factory=lambda: [1.0])  # Row sampling (0.6, 0.8, 1.0)
    bagging_temp_list: List[float] = field(default_factory=lambda: [1.0])  # Bayesian bootstrap intensity

    # CatBoost tree structure
    grow_policy_list: List[str] = field(default_factory=lambda: ['SymmetricTree'])  # SymmetricTree, Depthwise, Lossguide
    border_count_list: List[int] = field(default_factory=lambda: [254])  # Splits for numerical features (32, 128, 254)

    # Early stopping
    od_wait_list: List[int] = field(default_factory=lambda: [50])  # Early stopping patience (20, 50, 100)

    # Items to train
    item_ids: List[int] = field(default_factory=list)
    num_random_items: int = 0  # If > 0, select random items

    # Execution options
    parallel_variants: int = 1  # How many variants to run in parallel
    skip_existing: bool = True  # Skip variants that already have results


QUICK_SWEEP = SweepConfig(
    name="quick_sweep",
    description="Quick sweep of key configurations",
    training_days=[7, 30, 90],
    iterations_list=[200],
    depth_list=[6],
    learning_rate_list=[0.1],
    num_random_items=30,
)

WINDOW_SWEEP = SweepConfig(
    name="window_sweep",
    description="Training window comparison (main experiment)",
    training_days=[7, 14, 30, 60, 90, 180],
    iterations_list=[200],
    depth_list=[6],
    learning_rate_list=[0.1],
    num_random_items=50,
)

CATBOOST_SWEEP = SweepConfig(
    name="catboost_sweep",
    description="CatBoost hyperparameter search (core params)",
    training_days=[30],  # Fix window at 30 days
    iterations_list=[100, 200, 400],
    depth_list=[4, 6, 8],
    learning_rate_list=[0.05, 0.1, 0.2],
    num_random_items=30,
)

REGULARIZATION_SWEEP = SweepConfig(
    name="regularization_sweep",
    description="CatBoost regularization and sampling parameters",
    training_days=[30],
    iterations_list=[200],
    depth_list=[6],
    learning_rate_list=[0.1],
    l2_leaf_reg_list=[1.0, 3.0, 10.0],
    min_data_in_leaf_list=[1, 5, 10],
    subsample_list=[0.6, 0.8, 1.0],
    num_random_items=30,
)

TREE_STRUCTURE_SWEEP = SweepConfig(
    name="tree_structure_sweep",
    description="CatBoost tree growing strategies",
    training_days=[30],
    iterations_list=[200],
    depth_list=[6, 8],  # Max depth 8 for 12GB VRAM (RTX 3060)
    learning_rate_list=[0.1],
    grow_policy_list=['SymmetricTree', 'Depthwise', 'Lossguide'],
    border_count_list=[32, 128, 254],
    num_random_items=30,
)

CATEGORY_SWEEP = SweepConfig(
    name="category_sweep",
    description="Compare performance across item categories",
    training_days=[7, 30, 90],
    iterations_list=[200],
    depth_list=[6],
    learning_rate_list=[0.1],
    item_ids=HIGH_VOLUME_ITEMS + HIGH_VALUE_ITEMS + CONSUMABLE_ITEMS + EQUIPMENT_ITEMS,
)

FULL_SWEEP = SweepConfig(
    name="full_sweep",
    description="Full hyperparameter sweep (all combinations)",
    training_days=[7, 14, 30, 60, 90, 180],
    iterations_list=[100, 200, 400],
    depth_list=[4, 6, 8],
    learning_rate_list=[0.05, 0.1, 0.2],
    num_random_items=50,
)


# =============================================================================
# Sweep Execution
# =============================================================================

def get_sweep_items(config: SweepConfig) -> List[int]:
    """Get items for the sweep, either from config or randomly selected."""
    if config.item_ids:
        return config.item_ids

    if config.num_random_items > 0:
        with get_db_cursor() as cur:
            # Select high-volume items with active models for baseline comparison
            cur.execute("""
                WITH active_items AS (
                    SELECT DISTINCT item_id
                    FROM model_registry
                    WHERE status = 'ACTIVE'
                ),
                volume_ranked AS (
                    SELECT
                        p.item_id,
                        SUM(high_price_volume + low_price_volume) as volume
                    FROM price_data_5min p
                    WHERE p.timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY p.item_id
                )
                SELECT v.item_id
                FROM volume_ranked v
                JOIN active_items a ON v.item_id = a.item_id
                ORDER BY v.volume DESC
                LIMIT %s
            """, (config.num_random_items,))

            items = [row[0] for row in cur.fetchall()]

        logger.info(f"Selected {len(items)} items by volume")
        return items

    raise ValueError("No items specified for sweep")


def create_sweep_experiment(config: SweepConfig) -> str:
    """Create experiment record for the sweep."""
    experiment_id = f"sweep_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Calculate total variants
    total_variants = (
        len(config.training_days) *
        len(config.iterations_list) *
        len(config.depth_list) *
        len(config.learning_rate_list)
    )

    with get_db_cursor() as cur:
        cur.execute("""
            INSERT INTO experiments (experiment_id, name, description, config, status)
            VALUES (%s, %s, %s, %s, 'PENDING')
            RETURNING experiment_id
        """, (
            experiment_id,
            config.name,
            f"{config.description}\n\nTotal variants: {total_variants}",
            json.dumps(asdict(config), default=str)
        ))

    logger.info(f"Created sweep experiment: {experiment_id}")
    logger.info(f"Total variants to test: {total_variants}")

    return experiment_id


def generate_variants(config: SweepConfig) -> List[Dict[str, Any]]:
    """Generate all variant configurations for the sweep."""
    variants = []

    # Build list of all parameter combinations
    param_combos = list(product(
        config.training_days,
        config.iterations_list,
        config.depth_list,
        config.learning_rate_list,
        config.l2_leaf_reg_list,
        config.min_data_in_leaf_list,
        config.subsample_list,
        config.bagging_temp_list,
        config.grow_policy_list,
        config.border_count_list,
        config.od_wait_list,
    ))

    for combo in param_combos:
        (days, iters, depth, lr, l2_reg, min_leaf,
         subsample, bagging_temp, grow_policy, border_count, od_wait) = combo

        # Build variant name (only include non-default params)
        name_parts = [f"d{days}"]
        if iters != 200:
            name_parts.append(f"i{iters}")
        if depth != 6:
            name_parts.append(f"dp{depth}")
        if lr != 0.1:
            name_parts.append(f"lr{str(lr).replace('.', '')}")
        if l2_reg != 3.0:
            name_parts.append(f"l2{str(l2_reg).replace('.', '')}")
        if min_leaf != 1:
            name_parts.append(f"ml{min_leaf}")
        if subsample != 1.0:
            name_parts.append(f"ss{str(subsample).replace('.', '')}")
        if grow_policy != 'SymmetricTree':
            name_parts.append(grow_policy[:4].lower())
        if border_count != 254:
            name_parts.append(f"bc{border_count}")

        variant_name = "_".join(name_parts)

        variants.append({
            'name': variant_name,
            'training_days': days,
            'iterations': iters,
            'depth': depth,
            'learning_rate': lr,
            'l2_leaf_reg': l2_reg,
            'min_data_in_leaf': min_leaf,
            'subsample': subsample,
            'bagging_temperature': bagging_temp,
            'grow_policy': grow_policy,
            'border_count': border_count,
            'od_wait': od_wait,
        })

    return variants


def run_variant(
    experiment_id: str,
    variant: Dict[str, Any],
    item_ids: List[int],
    dry_run: bool = False
) -> Optional[str]:
    """
    Run a single variant of the sweep.

    Returns the run_id if successful, None otherwise.
    """
    variant_name = variant['name']
    run_id = f"{experiment_id}_{variant_name}"

    logger.info(f"\n{'='*60}")
    logger.info(f"VARIANT: {variant_name}")
    logger.info(f"  Days: {variant['training_days']}")
    logger.info(f"  Iterations: {variant['iterations']}")
    logger.info(f"  Depth: {variant['depth']}")
    logger.info(f"  Learning Rate: {variant['learning_rate']}")
    logger.info(f"  Items: {len(item_ids)}")
    logger.info(f"{'='*60}")

    # Create variant record
    with get_db_cursor() as cur:
        cur.execute("""
            INSERT INTO experiment_variants
                (experiment_id, variant_name, config, run_id, item_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id, variant_name) DO UPDATE
            SET run_id = EXCLUDED.run_id, item_count = EXCLUDED.item_count
        """, (
            experiment_id,
            variant_name,
            json.dumps(variant),
            run_id,
            len(item_ids)
        ))

    if dry_run:
        logger.info(f"[DRY RUN] Would train variant {variant_name}")
        return run_id

    try:
        # Step 1: Prepare data
        logger.info(f"Preparing training data ({variant['training_days']} days)...")
        items_arg = ','.join(str(i) for i in item_ids)

        prep_cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'cloud' / 'prepare_runpod_data.py'),
            '--days', str(variant['training_days']),
            '--items', items_arg,
            '--run-id', run_id,
            '--output-dir', f'/tmp/gept_sweep/{run_id}',
        ]

        result = subprocess.run(prep_cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"Data prep failed: {result.stderr}")
            return None

        # Step 2: Train models
        logger.info(f"Training {len(item_ids)} items...")

        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'cloud' / 'train_runpod_multitarget.py'),
            '--run-id', run_id,
            '--all',
            '--local',
            '--cache-dir', f'/tmp/gept_sweep/{run_id}',
            '--output-dir', f'{PROJECT_ROOT}/models/{run_id}',
            # CatBoost hyperparameters
            '--iterations', str(variant['iterations']),
            '--depth', str(variant['depth']),
            '--learning-rate', str(variant['learning_rate']),
            '--l2-leaf-reg', str(variant['l2_leaf_reg']),
            '--min-data-in-leaf', str(variant['min_data_in_leaf']),
            '--subsample', str(variant['subsample']),
            '--bagging-temperature', str(variant['bagging_temperature']),
            '--grow-policy', variant['grow_policy'],
            '--border-count', str(variant['border_count']),
            '--od-wait', str(variant['od_wait']),
        ]

        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=14400)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return None

        # Step 3: Collect results
        collect_variant_results(experiment_id, variant_name, run_id)

        logger.info(f"Variant {variant_name} complete")
        return run_id

    except subprocess.TimeoutExpired:
        logger.error(f"Variant {variant_name} timed out")
        return None
    except Exception as e:
        logger.error(f"Variant {variant_name} failed: {e}")
        return None


def collect_variant_results(experiment_id: str, variant_name: str, run_id: str):
    """Collect and store AUC results for a variant."""
    # Read training summary
    summary_file = PROJECT_ROOT / 'models' / run_id / 'training_summary.json'

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        mean_auc = summary.get('mean_auc')
        auc_std = summary.get('auc_std')

        with get_db_cursor() as cur:
            cur.execute("""
                UPDATE experiment_variants
                SET
                    mean_auc = %s,
                    std_auc = %s,
                    completed_at = NOW()
                WHERE experiment_id = %s AND variant_name = %s
            """, (mean_auc, auc_std, experiment_id, variant_name))

        logger.info(f"  Results: mean_auc={mean_auc:.4f}, std={auc_std:.4f}")


def analyze_sweep(experiment_id: str) -> Dict:
    """Analyze sweep results and identify best configuration."""
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT
                variant_name,
                config,
                mean_auc,
                std_auc,
                item_count
            FROM experiment_variants
            WHERE experiment_id = %s
              AND mean_auc IS NOT NULL
            ORDER BY mean_auc DESC
        """, (experiment_id,))

        results = []
        for row in cur.fetchall():
            results.append({
                'variant': row[0],
                'config': row[1],
                'mean_auc': float(row[2]) if row[2] else None,
                'std_auc': float(row[3]) if row[3] else None,
                'items': row[4],
            })

    if not results:
        logger.warning("No results found for sweep")
        return {'results': [], 'best': None, 'analysis': None}

    best = results[0]

    # Analyze by dimension
    analysis = {
        'by_training_days': {},
        'by_iterations': {},
        'by_depth': {},
        'by_learning_rate': {},
    }

    for r in results:
        cfg = r['config']
        auc = r['mean_auc']
        if auc is None:
            continue

        days = cfg.get('training_days')
        if days:
            if days not in analysis['by_training_days']:
                analysis['by_training_days'][days] = []
            analysis['by_training_days'][days].append(auc)

        iters = cfg.get('iterations')
        if iters:
            if iters not in analysis['by_iterations']:
                analysis['by_iterations'][iters] = []
            analysis['by_iterations'][iters].append(auc)

        depth = cfg.get('depth')
        if depth:
            if depth not in analysis['by_depth']:
                analysis['by_depth'][depth] = []
            analysis['by_depth'][depth].append(auc)

        lr = cfg.get('learning_rate')
        if lr:
            if lr not in analysis['by_learning_rate']:
                analysis['by_learning_rate'][lr] = []
            analysis['by_learning_rate'][lr].append(auc)

    # Compute averages
    for dim in analysis:
        for key in analysis[dim]:
            vals = analysis[dim][key]
            analysis[dim][key] = {
                'mean': sum(vals) / len(vals),
                'count': len(vals),
            }

    # Print results
    logger.info("\n" + "="*70)
    logger.info("SWEEP RESULTS")
    logger.info("="*70)

    logger.info(f"\nBest configuration: {best['variant']}")
    logger.info(f"  Mean AUC: {best['mean_auc']:.4f}")
    logger.info(f"  Config: {json.dumps(best['config'], indent=4)}")

    logger.info("\n--- By Training Window ---")
    for days in sorted(analysis['by_training_days'].keys()):
        info = analysis['by_training_days'][days]
        logger.info(f"  {days:3d} days: AUC={info['mean']:.4f} (n={info['count']})")

    if len(analysis['by_iterations']) > 1:
        logger.info("\n--- By Iterations ---")
        for iters in sorted(analysis['by_iterations'].keys()):
            info = analysis['by_iterations'][iters]
            logger.info(f"  {iters:4d}: AUC={info['mean']:.4f} (n={info['count']})")

    if len(analysis['by_depth']) > 1:
        logger.info("\n--- By Depth ---")
        for depth in sorted(analysis['by_depth'].keys()):
            info = analysis['by_depth'][depth]
            logger.info(f"  {depth:2d}: AUC={info['mean']:.4f} (n={info['count']})")

    if len(analysis['by_learning_rate']) > 1:
        logger.info("\n--- By Learning Rate ---")
        for lr in sorted(analysis['by_learning_rate'].keys()):
            info = analysis['by_learning_rate'][lr]
            logger.info(f"  {lr:.2f}: AUC={info['mean']:.4f} (n={info['count']})")

    logger.info("="*70)

    return {
        'results': results,
        'best': best,
        'analysis': analysis,
    }


def run_sweep(config: SweepConfig, dry_run: bool = False) -> str:
    """Run the full hyperparameter sweep."""
    logger.info("="*70)
    logger.info(f"HYPERPARAMETER SWEEP: {config.name}")
    logger.info("="*70)
    logger.info(config.description)
    logger.info("="*70)

    # Get items
    item_ids = get_sweep_items(config)
    logger.info(f"Items to train: {len(item_ids)}")

    # Create experiment
    experiment_id = create_sweep_experiment(config)

    # Update status
    with get_db_cursor() as cur:
        cur.execute("""
            UPDATE experiments
            SET status = 'RUNNING', started_at = NOW()
            WHERE experiment_id = %s
        """, (experiment_id,))

    # Generate variants
    variants = generate_variants(config)
    logger.info(f"Variants to test: {len(variants)}")

    # Run each variant
    completed = 0
    failed = 0

    for i, variant in enumerate(variants):
        logger.info(f"\n[{i+1}/{len(variants)}] Running variant: {variant['name']}")

        run_id = run_variant(experiment_id, variant, item_ids, dry_run)

        if run_id:
            completed += 1
        else:
            failed += 1

    # Finalize
    with get_db_cursor() as cur:
        cur.execute("""
            UPDATE experiments
            SET status = 'COMPLETED', completed_at = NOW()
            WHERE experiment_id = %s
        """, (experiment_id,))

    logger.info(f"\nSweep complete: {completed} succeeded, {failed} failed")

    # Analyze results
    if not dry_run:
        analyze_sweep(experiment_id)

    return experiment_id


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter sweep for GePT models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Sweep presets
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument('--quick', action='store_true',
                             help='Quick sweep (3 windows, 30 items)')
    sweep_group.add_argument('--windows-only', action='store_true',
                             help='Training window sweep only')
    sweep_group.add_argument('--catboost-only', action='store_true',
                             help='CatBoost core hyperparameter sweep')
    sweep_group.add_argument('--regularization', action='store_true',
                             help='Regularization and sampling sweep')
    sweep_group.add_argument('--tree-structure', action='store_true',
                             help='Tree growing strategy sweep')
    sweep_group.add_argument('--category', action='store_true',
                             help='Category comparison sweep')
    sweep_group.add_argument('--full', action='store_true',
                             help='Full sweep (all combinations)')

    # Custom parameters (override preset)
    parser.add_argument('--days', type=int, nargs='+',
                        help='Training window days to test')
    parser.add_argument('--iterations', type=int, nargs='+',
                        help='CatBoost iterations to test')
    parser.add_argument('--depth', type=int, nargs='+',
                        help='CatBoost depth to test')
    parser.add_argument('--learning-rate', type=float, nargs='+',
                        help='Learning rates to test')
    parser.add_argument('--items', type=int, default=None,
                        help='Number of items to train per variant')

    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without running')
    parser.add_argument('--analyze', type=str, metavar='EXPERIMENT_ID',
                        help='Analyze existing sweep results')

    args = parser.parse_args()

    # Analyze existing sweep
    if args.analyze:
        results = analyze_sweep(args.analyze)
        print(json.dumps(results, indent=2, default=str))
        return

    # Select preset
    if args.full:
        config = FULL_SWEEP
    elif args.catboost_only:
        config = CATBOOST_SWEEP
    elif args.regularization:
        config = REGULARIZATION_SWEEP
    elif args.tree_structure:
        config = TREE_STRUCTURE_SWEEP
    elif args.category:
        config = CATEGORY_SWEEP
    elif args.windows_only:
        config = WINDOW_SWEEP
    else:
        config = QUICK_SWEEP

    # Override with custom parameters
    if args.days:
        config.training_days = args.days
    if args.iterations:
        config.iterations_list = args.iterations
    if args.depth:
        config.depth_list = args.depth
    if args.learning_rate:
        config.learning_rate_list = args.learning_rate
    if args.items:
        config.num_random_items = args.items
        config.item_ids = []  # Clear preset items

    # Run sweep
    experiment_id = run_sweep(config, dry_run=args.dry_run)

    print(f"\nExperiment ID: {experiment_id}")
    print(f"Analyze with: python {__file__} --analyze {experiment_id}")


if __name__ == '__main__':
    main()
