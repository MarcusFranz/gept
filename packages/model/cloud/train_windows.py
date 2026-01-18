#!/usr/bin/env python3
"""
Optimized CatBoost training script for Windows/WSL2 primary training server.

Target machine: i7-10700f (8 cores / 16 threads), 32GB RAM, RTX 3060 12GB

Performance optimizations:
- No StandardScaler (CatBoost handles numeric features natively)
- Logloss for early stopping (faster than AUC)
- boosting_type='Plain' (faster than Ordered)
- High parallelism (8+ workers × 2 threads each)

Usage:
    # Full training
    python train_windows.py --all --workers 8

    # Tier-based training
    python train_windows.py --tier A --workers 8

    # Specific items
    python train_windows.py --items 10006,10008,10010 --workers 8

    # Resume interrupted run
    python train_windows.py --all --resume --workers 8

    # Fast mode (trades ~0.01 AUC for speed)
    python train_windows.py --all --workers 8 --fast

    # GPU mode (for benchmarking)
    python train_windows.py --all --gpu --workers 1

    # Benchmark configurations
    python train_windows.py --items 10006,10008 --benchmark
"""
from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

# Add src to path for centralized config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from training_tier_config import get_tier_from_auc  # noqa: E402
from catboost_config import load_catboost_params  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("ERROR: CatBoost not installed. Run: pip install catboost")
    sys.exit(1)

try:
    import skl2onnx  # noqa: F401 - used for availability check
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("WARNING: ONNX not installed. Models will be saved as .cbm only.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================================
# Configuration
# ============================================================================

# Load centralized CatBoost params (Issue #69)
_BASE_CATBOOST_CONFIG = load_catboost_params()

# Base CatBoost params (optimized for throughput with minimal quality impact)
# Core hyperparameters loaded from config/training_config.yaml
CATBOOST_PARAMS = {
    'iterations': _BASE_CATBOOST_CONFIG.get('iterations', 200),
    'depth': _BASE_CATBOOST_CONFIG.get('depth', 6),
    'learning_rate': _BASE_CATBOOST_CONFIG.get('learning_rate', 0.1),
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',  # Faster than AUC for early stopping
    'random_state': 42,
    'verbose': 0,
    'task_type': 'CPU',
    'thread_count': 2,  # Low per-model, high parallelism across items
    'boosting_type': 'Plain',  # Faster than Ordered
    'early_stopping_rounds': _BASE_CATBOOST_CONFIG.get('od_wait', 20),
    'use_best_model': True,
}

# Additional params for --fast mode
FAST_MODE_PARAMS = {
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'rsm': 0.8,  # Column sampling
    'min_data_in_leaf': 30,
}

# GPU mode params
GPU_PARAMS = {
    'task_type': 'GPU',
    'devices': '0',
}

# Default paths
DEFAULT_BUCKET = 'osrs-models-mof'
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'gept' / 'data'
DEFAULT_OUTPUT_DIR = Path.home() / 'models'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading with Caching
# ============================================================================

def get_cache_path(cache_dir: Path, run_id: str, filename: str) -> Path:
    """Get path for cached file."""
    return cache_dir / run_id / filename


def load_config_cached(bucket_name: str, run_id: str, cache_dir: Path) -> dict:
    """Load run config from GCS with local caching."""
    cache_path = get_cache_path(cache_dir, run_id, 'config.json')

    # Check cache first
    if cache_path.exists():
        logger.debug(f"Loading config from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    # Download from GCS
    logger.info(f"Downloading config from gs://{bucket_name}/runs/{run_id}/config.json")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'runs/{run_id}/config.json')
    config = json.loads(blob.download_as_string())

    # Cache locally
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(config, f)

    return config


def load_tier_state_cached(bucket_name: str, cache_dir: Path) -> dict:
    """Load tier state from GCS with local caching."""
    cache_path = cache_dir / 'tier_state.json'

    # For tier state, always refresh from GCS (it changes frequently)
    # but keep a fallback cache
    try:
        logger.debug(f"Downloading tier state from gs://{bucket_name}/tier_state.json")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob('tier_state.json')
        tier_state = json.loads(blob.download_as_string())

        # Update cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(tier_state, f)

        return tier_state

    except Exception as e:
        logger.warning(f"Could not load tier state from GCS: {e}")

        # Try cache
        if cache_path.exists():
            logger.info("Using cached tier state")
            with open(cache_path) as f:
                return json.load(f)

        # Return empty state
        return {'items': {}, 'tier_counts': {'A': 0, 'B': 0, 'C': 0, 'D': 0}}


def load_item_data_cached(
    bucket_name: str,
    run_id: str,
    item_id: int,
    cache_dir: Path
) -> pd.DataFrame:
    """Load item data from GCS with local caching."""
    cache_path = get_cache_path(cache_dir, run_id, f'{item_id}.parquet')

    # Check cache first
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return pd.read_parquet(cache_path)

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'runs/{run_id}/data/{item_id}.parquet')
    data = blob.download_as_bytes()

    # Cache locally
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        f.write(data)

    # Read and return
    table = pq.read_table(BytesIO(data))
    return table.to_pandas()


# ============================================================================
# Item Filtering
# ============================================================================

def filter_items(
    items: list[dict],
    tier: Optional[str],
    item_ids: Optional[list[int]],
    tier_state: dict
) -> list[dict]:
    """Filter items based on tier or specific item IDs."""
    if item_ids:
        # Filter by specific item IDs
        item_id_set = set(item_ids)
        return [i for i in items if i['item_id'] in item_id_set]

    if tier:
        # Filter by tier
        tier_items = tier_state.get('items', {})
        return [
            i for i in items
            if tier_items.get(str(i['item_id']), {}).get('tier', 'D') == tier
        ]

    # Return all items
    return items


def is_already_trained(output_dir: Path, run_id: str, item_id: int) -> bool:
    """Check if item has already been trained (registry.json exists)."""
    registry_path = output_dir / run_id / str(item_id) / 'registry.json'
    return registry_path.exists()


# ============================================================================
# Training Core
# ============================================================================

def train_single_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict
) -> tuple[Optional[CatBoostClassifier], dict]:
    """Train a single CatBoost model."""
    # Check minimum samples
    if y_train.sum() < 10 or y_test.sum() < 5:
        return None, {'status': 'skipped', 'reason': 'insufficient_positive_samples'}

    try:
        # Train model (CatBoost handles numeric features natively, no scaler needed)
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

        # Compute AUC (only at end, not during early stopping)
        y_pred = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_pred)
        except ValueError:
            auc = 0.5

        metrics = {
            'status': 'success',
            'auc': float(auc),
            'positive_rate_train': float(y_train.mean()),
            'positive_rate_test': float(y_test.mean()),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'is_valid': auc > 0.52,
            'iterations_used': model.get_best_iteration() or params.get('iterations', 200)
        }

        return model, metrics

    except Exception as e:
        return None, {'status': 'error', 'error': str(e)}


def train_item(
    item_config: dict,
    df: pd.DataFrame,
    params: dict
) -> dict:
    """Train all models for a single item."""
    item_id = item_config['item_id']
    item_name = item_config.get('item_name', str(item_id))
    feature_cols = item_config['feature_cols']
    target_cols = item_config['target_cols']

    start_time = time.time()

    # Prepare features (NO StandardScaler - CatBoost handles numeric features natively)
    X = df[feature_cols].values

    # Time-based train/test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]

    models = {}
    aucs = []

    for target_col in target_cols:
        y = df[target_col].values
        y_train, y_test = y[:train_size], y[train_size:]

        model, metrics = train_single_model(X_train, X_test, y_train, y_test, params)

        if model is not None and metrics.get('is_valid', False):
            models[target_col] = {'model': model, 'metrics': metrics}
            aucs.append(metrics['auc'])

    elapsed = time.time() - start_time
    avg_auc = np.mean(aucs) if aucs else 0.5
    tier = get_tier_from_auc(avg_auc)

    return {
        'item_id': item_id,
        'item_name': item_name,
        'models': models,
        'models_trained': len(target_cols),
        'models_valid': len(models),
        'avg_auc': avg_auc,
        'tier': tier,
        'elapsed': elapsed,
        'feature_cols': feature_cols,
        'n_rows': len(df),
    }


# ============================================================================
# Model Saving
# ============================================================================

def export_model_to_onnx(model: CatBoostClassifier, n_features: int) -> Optional[bytes]:
    """Export CatBoost model to ONNX format."""
    if not HAS_ONNX:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            model.save_model(tmp.name, format='onnx')
        with open(tmp.name, 'rb') as f:
            data = f.read()
        os.unlink(tmp.name)
        return data
    except Exception as e:
        logger.debug(f"ONNX export failed: {e}")
        return None


def save_models_local(output_dir: Path, run_id: str, results: dict):
    """Save trained models to local filesystem."""
    item_id = results['item_id']
    timestamp = datetime.now().isoformat()
    n_features = len(results.get('feature_cols', []))

    # Create item directory
    item_dir = output_dir / run_id / str(item_id)
    item_dir.mkdir(parents=True, exist_ok=True)

    # Save each model
    for target_name, model_data in results.get('models', {}).items():
        model = model_data['model']

        # Save native CatBoost format (.cbm)
        cbm_path = item_dir / f'{target_name}.cbm'
        model.save_model(str(cbm_path))

        # Save ONNX format
        if HAS_ONNX:
            onnx_data = export_model_to_onnx(model, n_features)
            if onnx_data:
                onnx_path = item_dir / f'{target_name}.onnx'
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_data)

        # Save metadata
        meta = {
            'auc': model_data['metrics']['auc'],
            'trained_at': timestamp,
            'iterations_used': model_data['metrics'].get('iterations_used', 200),
        }
        meta_path = item_dir / f'{target_name}_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    # Save registry
    registry = {
        'item_id': item_id,
        'item_name': results.get('item_name', ''),
        'trained_at': timestamp,
        'models_trained': results['models_trained'],
        'models_valid': results['models_valid'],
        'avg_auc': results['avg_auc'],
        'tier': results['tier'],
        'elapsed': results['elapsed'],
        'n_rows': results.get('n_rows', 0),
        'onnx_enabled': HAS_ONNX,
    }
    registry_path = item_dir / 'registry.json'
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


# ============================================================================
# Worker Function (for ProcessPoolExecutor)
# ============================================================================

def process_item(args: tuple) -> dict:
    """Process a single item (runs in worker process)."""
    bucket_name, run_id, item_config, output_dir, cache_dir, params = args
    item_id = item_config['item_id']
    item_name = item_config.get('item_name', str(item_id))

    try:
        # Load data (with caching)
        df = load_item_data_cached(bucket_name, run_id, item_id, Path(cache_dir))

        # Train models
        results = train_item(item_config, df, params)

        # Save models
        save_models_local(Path(output_dir), run_id, results)

        return {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'success',
            'models_valid': results['models_valid'],
            'avg_auc': results['avg_auc'],
            'tier': results['tier'],
            'elapsed': results['elapsed'],
            'n_rows': results.get('n_rows', 0),
        }

    except Exception as e:
        import traceback
        return {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


# ============================================================================
# Benchmark Mode
# ============================================================================

def run_benchmark(
    items: list[dict],
    bucket_name: str,
    run_id: str,
    cache_dir: Path,
    output_dir: Path
):
    """Run benchmark comparing different configurations."""
    logger.info("=" * 60)
    logger.info("BENCHMARK MODE")
    logger.info("=" * 60)

    # Limit to first 2 items, 5 targets each
    benchmark_items = items[:2]
    for item in benchmark_items:
        item['target_cols'] = item['target_cols'][:5]

    configs = {
        'baseline': dict(CATBOOST_PARAMS),
        'plain_boost': {**CATBOOST_PARAMS, 'boosting_type': 'Plain'},
        'fast_mode': {**CATBOOST_PARAMS, **FAST_MODE_PARAMS},
    }

    # Add GPU config if available
    try:
        test_model = CatBoostClassifier(task_type='GPU', iterations=1, verbose=0)
        test_model.fit([[0, 1], [1, 0]], [0, 1])
        configs['gpu_mode'] = {**CATBOOST_PARAMS, **GPU_PARAMS, 'thread_count': None}
        logger.info("GPU available, including in benchmark")
    except Exception:
        logger.info("GPU not available, skipping GPU benchmark")

    results = {}

    for config_name, params in configs.items():
        logger.info(f"\nBenchmarking config: {config_name}")
        config_results = []

        for item_config in benchmark_items:
            item_id = item_config['item_id']
            df = load_item_data_cached(bucket_name, run_id, item_id, cache_dir)

            start = time.time()
            train_results = train_item(item_config, df, params)
            elapsed = time.time() - start

            config_results.append({
                'item_id': item_id,
                'avg_auc': train_results['avg_auc'],
                'models_valid': train_results['models_valid'],
                'elapsed': elapsed,
            })

            logger.info(
                f"  Item {item_id}: AUC={train_results['avg_auc']:.4f}, "
                f"models={train_results['models_valid']}, time={elapsed:.2f}s"
            )

        results[config_name] = {
            'items': config_results,
            'avg_auc': np.mean([r['avg_auc'] for r in config_results]),
            'total_time': sum(r['elapsed'] for r in config_results),
        }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    baseline_auc = results.get('baseline', {}).get('avg_auc', 0.5)
    baseline_time = results.get('baseline', {}).get('total_time', 1)

    for config_name, config_results in results.items():
        auc_delta = config_results['avg_auc'] - baseline_auc
        speedup = baseline_time / config_results['total_time'] if config_results['total_time'] > 0 else 0
        logger.info(
            f"{config_name:15s}: AUC={config_results['avg_auc']:.4f} "
            f"(delta={auc_delta:+.4f}), time={config_results['total_time']:.2f}s "
            f"(speedup={speedup:.2f}x)"
        )

    # Save results
    benchmark_path = output_dir / 'benchmark_results.json'
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {benchmark_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimized CatBoost training for Windows/WSL2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_windows.py --all --workers 8
  python train_windows.py --tier A --workers 8
  python train_windows.py --items 10006,10008,10010 --workers 8
  python train_windows.py --all --resume --workers 8
  python train_windows.py --all --fast --workers 8
  python train_windows.py --all --gpu --workers 1
  python train_windows.py --items 10006,10008 --benchmark
        """
    )

    # Item selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Train all items')
    group.add_argument('--tier', choices=['A', 'B', 'C', 'D'], help='Train items of specific tier')
    group.add_argument('--items', type=str, help='Comma-separated item IDs (e.g., 10006,10008)')
    group.add_argument('--benchmark', action='store_true', help='Run benchmark mode')

    # Parallelism
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    parser.add_argument('--threads-per-model', type=int, default=2, help='CatBoost threads per model (default: 2)')

    # Training modes
    parser.add_argument('--fast', action='store_true', help='Enable fast mode (trades ~0.01 AUC for speed)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training (1 model at a time)')

    # Resume/force
    parser.add_argument('--resume', action='store_true', help='Skip already-trained items')
    parser.add_argument('--force', action='store_true', help='Force retrain (ignore existing)')

    # Paths
    parser.add_argument('--bucket', default=DEFAULT_BUCKET, help=f'GCS bucket (default: {DEFAULT_BUCKET})')
    parser.add_argument('--run-id', required=True, help='Run ID from prepare_training_data.py')
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR, help=f'Local cache dir (default: {DEFAULT_CACHE_DIR})')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help=f'Output dir for models (default: {DEFAULT_OUTPUT_DIR})')

    # Misc
    parser.add_argument('--limit', type=int, help='Limit number of items (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("GePT Windows/WSL Training Script")
    logger.info("=" * 60)
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Threads per model: {args.threads_per_model}")
    logger.info(f"Fast mode: {args.fast}")
    logger.info(f"GPU mode: {args.gpu}")
    logger.info(f"ONNX enabled: {HAS_ONNX}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Build CatBoost params
    params = dict(CATBOOST_PARAMS)
    params['thread_count'] = args.threads_per_model

    if args.fast:
        params.update(FAST_MODE_PARAMS)
        logger.info("Fast mode enabled: using Bernoulli sampling, rsm=0.8, min_data_in_leaf=30")

    if args.gpu:
        params.update(GPU_PARAMS)
        params['thread_count'] = None  # Let CatBoost manage GPU threads
        logger.info("GPU mode enabled")

    # Load config and tier state
    logger.info("\nLoading configuration...")
    config = load_config_cached(args.bucket, args.run_id, args.cache_dir)
    tier_state = load_tier_state_cached(args.bucket, args.cache_dir)

    items = config['items']
    logger.info(f"Total items in config: {len(items)}")

    # Handle benchmark mode
    if args.benchmark:
        run_benchmark(items, args.bucket, args.run_id, args.cache_dir, args.output_dir)
        return

    # Parse item IDs if provided
    item_ids = None
    if args.items:
        item_ids = [int(x.strip()) for x in args.items.split(',')]

    # Filter items
    items = filter_items(items, args.tier, item_ids, tier_state)
    logger.info(f"Items after filtering: {len(items)}")

    # Apply limit
    if args.limit:
        items = items[:args.limit]
        logger.info(f"Items after limit: {len(items)}")

    # Apply resume logic
    if args.resume and not args.force:
        original_count = len(items)
        items = [
            i for i in items
            if not is_already_trained(args.output_dir, args.run_id, i['item_id'])
        ]
        skipped = original_count - len(items)
        if skipped > 0:
            logger.info(f"Skipped {skipped} already-trained items (--resume)")
        logger.info(f"Items to train: {len(items)}")

    if not items:
        logger.info("No items to train. Exiting.")
        return

    # Prepare work items
    work_items = [
        (args.bucket, args.run_id, item, str(args.output_dir), str(args.cache_dir), params)
        for item in items
    ]

    # Run training
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    start_time = time.time()
    completed = 0
    errors = []
    tier_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    # GPU mode: single worker (1 GPU model at a time)
    workers = 1 if args.gpu else args.workers

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_item, item): item for item in work_items}

        # Progress tracking
        if HAS_TQDM:
            futures_iter = tqdm(as_completed(futures), total=len(futures), desc="Training")
        else:
            futures_iter = as_completed(futures)

        for future in futures_iter:
            result = future.result()
            completed += 1

            if result['status'] == 'success':
                tier_counts[result['tier']] += 1
                if not HAS_TQDM:
                    logger.info(
                        f"[{completed}/{len(items)}] Item {result['item_id']} ({result['item_name']}): "
                        f"{result['models_valid']} models, AUC {result['avg_auc']:.4f}, "
                        f"Tier {result['tier']}, {result['elapsed']:.1f}s"
                    )
            else:
                errors.append(result)
                logger.error(
                    f"[{completed}/{len(items)}] Item {result['item_id']} ({result['item_name']}): "
                    f"ERROR - {result.get('error', 'unknown')}"
                )

            # Progress update every 50 items
            if not HAS_TQDM and completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed * 60
                remaining = (len(items) - completed) / rate if rate > 0 else 0
                logger.info(
                    f"    === Progress: {completed}/{len(items)}, {rate:.1f}/min, "
                    f"~{remaining:.0f}min remaining ==="
                )
                logger.info(
                    f"    === Tiers: A={tier_counts['A']}, B={tier_counts['B']}, "
                    f"C={tier_counts['C']}, D={tier_counts['D']} ==="
                )

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Completed: {completed - len(errors)}/{len(items)} items")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Rate: {(completed - len(errors))/total_time*60:.1f} items/minute")
    logger.info(
        f"Tier distribution: A={tier_counts['A']}, B={tier_counts['B']}, "
        f"C={tier_counts['C']}, D={tier_counts['D']}"
    )
    logger.info(f"Models saved to: {args.output_dir / args.run_id}")

    # Save training summary
    summary = {
        'run_id': args.run_id,
        'completed_at': datetime.now().isoformat(),
        'total_items': len(items),
        'successful': completed - len(errors),
        'errors': len(errors),
        'total_time_seconds': total_time,
        'items_per_minute': (completed - len(errors)) / total_time * 60 if total_time > 0 else 0,
        'tier_counts': tier_counts,
        'params': params,
        'fast_mode': args.fast,
        'gpu_mode': args.gpu,
    }
    summary_path = args.output_dir / args.run_id / 'training_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save errors if any
    if errors:
        errors_path = args.output_dir / args.run_id / 'errors.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"Errors saved to: {errors_path}")


if __name__ == '__main__':
    main()
