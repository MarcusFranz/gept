#!/usr/bin/env python3
"""
GePT Multi-Target Training Script

Trains CatBoost MultiLogloss models with 108 sequential fill targets:
- 18 time windows: [1-12h hourly, 16h, 20h, 24h, 32h, 40h, 48h]
- 6 offsets: [1.25%, 1.5%, 1.75%, 2%, 2.25%, 2.5%]

Features:
- Sequential fill targets (Issue #26: buy must fill before sell)
- MultiLogloss for efficient multi-target training
- Numba-accelerated target computation (parallel)
- I/O prefetching to overlap disk reads with GPU training
- Early stopping with validation monitoring
- Column-pruned parquet reads for memory efficiency
- Native .cbm model format

Performance Characteristics:
    GPU                 | VRAM  | ~Time/Model | ~400 Items
    --------------------|-------|-------------|------------
    RTX 6000 Ada        | 48GB  | 10-12s      | ~70 min
    RTX 4090            | 24GB  | 15-20s      | ~2 hours
    RTX 3060            | 12GB  | 40-50s      | ~5-6 hours
    CPU (16 cores)      | N/A   | 60-90s      | ~8-10 hours

Usage:
    # Full training (all items from data prep)
    python train_runpod_multitarget.py --run-id <run_id> --all --local \\
        --cache-dir /path/to/prepared/data --output-dir /path/to/models

    # High-end GPU (Ada/A100) - maximize throughput
    python train_runpod_multitarget.py --run-id <run_id> --all --local \\
        --threads 32 --numba-threads 16 --prefetch 4

    # Mid-range GPU (3060/4070) - balanced settings
    python train_runpod_multitarget.py --run-id <run_id> --all --local \\
        --threads 8 --numba-threads 8 --prefetch 2

    # Resume interrupted run (skips completed models)
    python train_runpod_multitarget.py --run-id <run_id> --all --resume

    # CPU mode (for testing or no GPU)
    python train_runpod_multitarget.py --run-id <run_id> --all --cpu --threads 16

Automation (for daily training pipeline):
    1. Run prepare_runpod_data.py first to generate run_id and prepared data
    2. Run this script with the run_id from step 1
    3. Models are saved to output-dir/<run_id>/<item_id>/model.cbm
    4. Check training_summary.json for success/error counts and AUC metrics
    5. Transfer successful models to production (see Issue #28 for lifecycle)

Environment Variables:
    DB_PASS             - Database password (for prepare_runpod_data.py)
    OMP_NUM_THREADS     - OpenMP threads (auto-set by --threads)
    NUMBA_NUM_THREADS   - Numba threads (auto-set by --numba-threads)

Requirements:
    pip install catboost numba pandas pyarrow scikit-learn tqdm

See Also:
    - prepare_runpod_data.py: Data preparation script (run first)
    - Issue #26: Sequential fill targets implementation
    - Issue #28: Automated daily training pipeline design
"""
from __future__ import annotations

import os
import sys

# Add src to path for centralized config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
import gc
import argparse
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Callable
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    print("WARNING: google-cloud-storage not installed. Using local data only.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("ERROR: CatBoost not installed. Run: pip install catboost")
    sys.exit(1)

try:
    from numba import njit, prange, set_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

USE_NUMBA = HAS_NUMBA

# Load centralized CatBoost parameters (Issue #69)
from catboost_config import load_catboost_params

# Training telemetry (Issue #75)
from training_metrics import TrainingMetrics

# Training data validation (Issue #77)
from training_validation import (
    validate_config,
    validate_training_data,
    validate_xy_shapes,
    ValidationError,
)

_CATBOOST_CONFIG = load_catboost_params()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for multi-target training."""
    # Time windows: [1-12h hourly] + [16h, 20h, 24h every 4h] + [32h, 40h, 48h every 8h]
    hours: List[int] = field(default_factory=lambda: [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  # Every hour to 12h
        16, 20, 24,  # Every 4h to 24h
        32, 40, 48   # Every 8h to 48h
    ])
    periods_per_hour: int = 12

    # Offsets: 1.25% to 2.5% in 0.25% increments
    offsets: List[float] = field(default_factory=lambda: [
        0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025
    ])

    # Data splits (temporal)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # CatBoost parameters (loaded from config/training_config.yaml - Issue #69)
    iterations: int = field(default_factory=lambda: _CATBOOST_CONFIG.get('iterations', 200))
    depth: int = field(default_factory=lambda: _CATBOOST_CONFIG.get('depth', 6))
    learning_rate: float = field(default_factory=lambda: _CATBOOST_CONFIG.get('learning_rate', 0.1))
    od_wait: int = field(default_factory=lambda: _CATBOOST_CONFIG.get('od_wait', 50))

    # Validation thresholds
    min_positive_samples: int = 20
    drop_first_n_rows: int = 300  # Feature warmup

    # Device
    use_gpu: bool = True
    cpu_threads: int = field(default_factory=lambda: max(1, os.cpu_count() or 4))
    numba_threads: Optional[int] = None
    gpu_ram_part: float = 0.85  # Leave 15% safety margin for other processes

    # GCS settings
    bucket_name: str = 'osrs-models-mof'

    @property
    def n_targets(self) -> int:
        return len(self.hours) * len(self.offsets)

    @property
    def target_names(self) -> List[str]:
        """Generate target names: seq_1h_1.25pct, seq_2h_1.25pct, ..."""
        names = []
        for offset in self.offsets:
            for hour in self.hours:
                pct_str = f"{offset*100:.2f}".rstrip('0').rstrip('.')
                names.append(f'seq_{hour}h_{pct_str}pct')
        return names


# Default paths
DEFAULT_BUCKET = 'osrs-models-mof'
DEFAULT_CACHE_DIR = Path('/root/data/prepared')  # Local data from prepare_runpod_data.py
DEFAULT_OUTPUT_DIR = Path('/workspace/models')  # Persistent volume on RunPod

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Credential Validation
# =============================================================================

def validate_gcs_credentials() -> None:
    """
    Validate GCS credentials at startup before any training begins.

    Raises:
        SystemExit: If credentials are missing, invalid, or authentication fails.
    """
    errors = []

    # Check GOOGLE_APPLICATION_CREDENTIALS environment variable
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        errors.append("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    elif not os.path.exists(creds_path):
        errors.append(f"Credentials file not found: {creds_path}")
    else:
        # Validate the credentials file is readable JSON
        try:
            with open(creds_path, 'r') as f:
                import json
                creds_data = json.load(f)
                # Check for required fields in service account key
                required_fields = ['type', 'project_id', 'private_key_id']
                missing_fields = [field for field in required_fields if field not in creds_data]
                if missing_fields:
                    errors.append(f"Credentials file missing required fields: {missing_fields}")
        except json.JSONDecodeError as e:
            errors.append(f"Credentials file is not valid JSON: {e}")
        except PermissionError:
            errors.append(f"Cannot read credentials file: {creds_path}")

    if errors:
        for error in errors:
            logger.error(f"GCS credential validation failed: {error}")
        logger.error("Please set GOOGLE_APPLICATION_CREDENTIALS to a valid service account key file")
        sys.exit(1)

    # Test actual GCS authentication
    if not HAS_GCS:
        logger.warning("google-cloud-storage not installed - skipping GCS auth test")
        return

    try:
        client = storage.Client()
        # Try to list buckets as a simple auth check (just fetches first page)
        list(client.list_buckets(max_results=1))
        logger.info("GCS credentials validated successfully")
    except Exception as e:
        logger.error(f"GCS authentication failed: {e}")
        logger.error("Check that your service account has appropriate permissions")
        sys.exit(1)


# =============================================================================
# Runtime Configuration
# =============================================================================

def configure_threading(cpu_threads: int, numba_threads: Optional[int]) -> None:
    """Configure CPU/Numba thread usage for maximum throughput."""
    if cpu_threads:
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            os.environ[var] = str(cpu_threads)
    if HAS_NUMBA:
        set_num_threads(numba_threads or cpu_threads)


def read_parquet_path(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read parquet from a local path with optional column pruning."""
    try:
        table = pq.read_table(path, columns=columns, use_threads=True)
        return table.to_pandas(use_threads=True, self_destruct=True)
    except Exception as exc:
        if columns:
            logger.warning(f"Column-pruned read failed for {path}: {exc}. Falling back to full read.")
            table = pq.read_table(path, use_threads=True)
            return table.to_pandas(use_threads=True, self_destruct=True)
        raise


def read_parquet_bytes(data: bytes, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read parquet from bytes with optional column pruning."""
    try:
        table = pq.read_table(BytesIO(data), columns=columns, use_threads=True)
        return table.to_pandas(use_threads=True, self_destruct=True)
    except Exception as exc:
        if columns:
            logger.warning(f"Column-pruned read failed for GCS bytes: {exc}. Falling back to full read.")
            table = pq.read_table(BytesIO(data), use_threads=True)
            return table.to_pandas(use_threads=True, self_destruct=True)
        raise

# =============================================================================
# Data Loading
# =============================================================================

def get_cache_path(cache_dir: Path, run_id: str, filename: str) -> Path:
    """Get path for cached file."""
    return cache_dir / run_id / filename


def load_config(
    bucket_name: str,
    run_id: str,
    cache_dir: Path,
    use_local: bool = False,
    gcs_bucket: Optional[object] = None
) -> dict:
    """
    Load run config from local cache or GCS.

    Args:
        bucket_name: GCS bucket name
        run_id: Run identifier
        cache_dir: Local cache directory
        use_local: If True, only use local data (from prepare_runpod_data.py)

    Returns:
        Config dict with items list
    """
    # Check for local config first (from prepare_runpod_data.py)
    local_config_path = cache_dir / run_id / 'config.json'
    if local_config_path.exists():
        logger.info(f"Loading config from local: {local_config_path}")
        with open(local_config_path) as f:
            config = json.load(f)
        # Validate config schema (Issue #77)
        validate_config(config)
        return config

    # Check GCS cache path
    gcs_cache_path = get_cache_path(cache_dir, run_id, 'config.json')
    if gcs_cache_path.exists():
        logger.debug(f"Loading config from GCS cache: {gcs_cache_path}")
        with open(gcs_cache_path) as f:
            config = json.load(f)
        # Validate config schema (Issue #77)
        validate_config(config)
        return config

    if use_local:
        raise RuntimeError(f"Local config not found at {local_config_path}")

    if not HAS_GCS:
        raise RuntimeError("GCS not available and no local config found")

    # Download from GCS
    logger.info(f"Downloading config from gs://{bucket_name}/runs/{run_id}/config.json")
    if gcs_bucket is None:
        client = storage.Client()
        gcs_bucket = client.bucket(bucket_name)
    blob = gcs_bucket.blob(f'runs/{run_id}/config.json')
    config = json.loads(blob.download_as_string())

    # Cache locally
    gcs_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gcs_cache_path, 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)

    # Validate config schema (Issue #77)
    validate_config(config)

    return config


def load_item_data(
    bucket_name: str,
    run_id: str,
    item_id: int,
    cache_dir: Path,
    use_local: bool = False,
    gcs_bucket: Optional[object] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load item data from local cache or GCS.

    Args:
        bucket_name: GCS bucket name
        run_id: Run identifier
        item_id: Item ID to load
        cache_dir: Local cache directory
        use_local: If True, only use local data

    Returns:
        DataFrame with item data
    """
    # Check for local data first (from prepare_runpod_data.py)
    local_path = cache_dir / run_id / f'{item_id}.parquet'
    if local_path.exists() and local_path.stat().st_size > 0:
        return read_parquet_path(local_path, columns=columns)

    # Check GCS cache path
    gcs_cache_path = get_cache_path(cache_dir, run_id, f'{item_id}.parquet')
    if gcs_cache_path.exists() and gcs_cache_path.stat().st_size > 0:
        return read_parquet_path(gcs_cache_path, columns=columns)

    if use_local:
        raise RuntimeError(f"Local data not found for item {item_id} at {local_path}")

    if not HAS_GCS:
        raise RuntimeError(f"GCS not available and no local data for item {item_id}")

    # Download from GCS
    logger.debug(f"Downloading data for item {item_id} from GCS...")
    if gcs_bucket is None:
        client = storage.Client()
        gcs_bucket = client.bucket(bucket_name)
    blob = gcs_bucket.blob(f'runs/{run_id}/data/{item_id}.parquet')
    data = blob.download_as_bytes()

    # Cache locally
    gcs_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gcs_cache_path, 'wb') as f:
        f.write(data)

    # Read and return
    return read_parquet_bytes(data, columns=columns)


# =============================================================================
# Sequential Fill Target Computation
# =============================================================================

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def sequential_fill_target_nb(
        low_vals: np.ndarray,
        high_vals: np.ndarray,
        lookforward: int,
        offset: float,
        out: np.ndarray
    ) -> None:
        n = low_vals.shape[0]
        out_len = out.shape[0]
        n_valid = n - lookforward + 1
        if n_valid > out_len:
            n_valid = out_len
        if n_valid <= 0:
            return

        for i in prange(n_valid):
            buy_price = low_vals[i] * (1.0 - offset)
            sell_price = high_vals[i] * (1.0 + offset)
            buy_idx = -1
            end = i + lookforward

            # Start at i+1 to avoid same-bar lookahead bias (Issue #32)
            for t in range(i + 1, end):
                if low_vals[t] <= buy_price:
                    buy_idx = t
                    break

            if buy_idx == -1:
                out[i] = 0
                continue

            hit = 0
            for t in range(buy_idx, end):
                if high_vals[t] >= sell_price:
                    hit = 1
                    break
            out[i] = hit


def sequential_fill_target_numpy(
    low_vals: np.ndarray,
    high_vals: np.ndarray,
    lookforward: int,
    offset: float,
    out: np.ndarray,
    chunk_size: int = 2048
) -> None:
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(low_vals)
    out_len = len(out)
    n_valid = n - lookforward + 1
    if n_valid > out_len:
        n_valid = out_len
    if n_valid <= 0:
        return

    # Exclude period 0 to avoid same-bar lookahead bias (Issue #32)
    # With lookforward=1, no future bars exist, so all outputs remain 0
    if lookforward <= 1:
        return

    # periods starts at 1 (first future bar after decision point)
    periods = np.arange(1, lookforward)

    for start in range(0, n_valid, chunk_size):
        end = min(n_valid, start + chunk_size)
        low_slice = low_vals[start:end + lookforward - 1]
        high_slice = high_vals[start:end + lookforward - 1]

        low_windows = sliding_window_view(low_slice, lookforward)
        high_windows = sliding_window_view(high_slice, lookforward)

        buy_price = low_vals[start:end] * (1.0 - offset)
        sell_price = high_vals[start:end] * (1.0 + offset)

        # Slice [:, 1:] excludes same-bar (period 0) from fill checks
        buy_fills_matrix = low_windows[:, 1:] <= buy_price[:, None]
        buy_ever_fills = buy_fills_matrix.any(axis=1)
        first_buy_periods = np.argmax(buy_fills_matrix, axis=1) + 1  # +1 to account for sliced window

        sell_fills_matrix = high_windows[:, 1:] >= sell_price[:, None]
        after_buy_mask = periods[None, :] >= first_buy_periods[:, None]
        sell_after_buy = (sell_fills_matrix & after_buy_mask).any(axis=1)

        out[start:end] = (buy_ever_fills & sell_after_buy).astype(np.uint8)


def compute_sequential_fill_target(
    low_vals: np.ndarray,
    high_vals: np.ndarray,
    hour: int,
    offset: float,
    periods_per_hour: int,
    out_len: int
) -> np.ndarray:
    """
    Compute sequential fill target (buy must fill before sell).
    """
    lookforward = hour * periods_per_hour
    out = np.zeros(out_len, dtype=np.uint8)

    if len(low_vals) <= lookforward or out_len <= 0:
        return out

    if USE_NUMBA and HAS_NUMBA:
        sequential_fill_target_nb(low_vals, high_vals, lookforward, offset, out)
    else:
        sequential_fill_target_numpy(low_vals, high_vals, lookforward, offset, out)

    return out


def compute_all_targets(
    df: pd.DataFrame,
    config: TrainingConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all 108 sequential fill targets.

    Returns:
        y_matrix: Shape (N, 108) target matrix
        target_names: List of target names
    """
    # Get price columns - handle potential duplicates from feature_engine
    low_col = df['low']
    if isinstance(low_col, pd.DataFrame):
        low_col = low_col.iloc[:, 0]

    high_col = df['high']
    if isinstance(high_col, pd.DataFrame):
        high_col = high_col.iloc[:, 0]

    low_vals = np.ascontiguousarray(low_col.values, dtype=np.float32)
    high_vals = np.ascontiguousarray(high_col.values, dtype=np.float32)

    max_lookforward = max(config.hours) * config.periods_per_hour
    n_valid = len(low_vals) - max_lookforward + 1
    if n_valid <= 0:
        return np.empty((0, config.n_targets), dtype=np.uint8), config.target_names

    targets = np.empty((n_valid, config.n_targets), dtype=np.uint8)
    target_names = []
    col_idx = 0

    # Compute targets for each offset × hour combination
    for offset in config.offsets:
        for hour in config.hours:
            y = compute_sequential_fill_target(
                low_vals,
                high_vals,
                hour,
                offset,
                config.periods_per_hour,
                n_valid
            )
            targets[:, col_idx] = y
            col_idx += 1

            pct_str = f"{offset*100:.2f}".rstrip('0').rstrip('.')
            target_names.append(f'seq_{hour}h_{pct_str}pct')
    return targets, target_names


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Return AUC if both classes present; otherwise None."""
    if y_true.size == 0:
        return None
    if np.all(y_true == y_true[0]):
        return None
    return float(roc_auc_score(y_true, y_prob))


# =============================================================================
# Multi-Target Trainer
# =============================================================================

class MultiTargetTrainer:
    """Trains CatBoost MultiLogloss models with 108 targets."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_cols_master: Optional[List[str]] = None

    def create_model(self) -> CatBoostClassifier:
        """Create CatBoost classifier with MultiLogloss."""
        params = {
            'loss_function': 'MultiLogloss',
            'iterations': self.config.iterations,
            'depth': self.config.depth,
            'learning_rate': self.config.learning_rate,
            'random_seed': 42,
            'thread_count': self.config.cpu_threads,
            'eval_metric': 'MultiLogloss',
            'od_type': 'Iter',
            'od_wait': self.config.od_wait,
            'use_best_model': True,
            'allow_writing_files': False,
            'verbose': 100,  # Show progress every 100 iterations
        }

        if self.config.use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
            params['gpu_ram_part'] = self.config.gpu_ram_part
            params['boosting_type'] = 'Plain'
        else:
            params['task_type'] = 'CPU'

        return CatBoostClassifier(**params)

    def set_feature_columns(self, feature_cols: List[str]) -> None:
        self.feature_cols_master = feature_cols

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names from dataframe."""
        feature_prefixes = [
            'return_', 'volatility_', 'volume_', 'ma_', 'spread',
            'rsi_', 'hour', 'momentum_', 'log_',
            # Critical additions to match FeatureEngine output:
            'range_', 'dist_', 'parkinson_', 'dow_', 'is_', 'mid_'
        ]

        # Also include some computed high/low features but NOT raw high/low
        high_low_features = ['high_N', 'low_N', 'high_low_range']

        feature_cols = []
        for col in df.columns:
            # Skip target columns (roundtrip_*, seq_*)
            if col.startswith('roundtrip_') or col.startswith('seq_'):
                continue
            # Skip raw price columns we use for targets
            if col in ['high', 'low', 'avg_high_price', 'avg_low_price']:
                continue
            # Include feature prefixes
            if any(col.startswith(p) for p in feature_prefixes):
                feature_cols.append(col)
            # Include specific high/low features
            if col in high_low_features:
                feature_cols.append(col)

        return sorted(list(set(feature_cols)))

    def temporal_split(self, n: int) -> Tuple[int, int]:
        """Get temporal split indices."""
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return train_end, val_end

    def train_item(
        self,
        item_id: int,
        item_name: str,
        df: pd.DataFrame,
        feature_cols_override: Optional[List[str]] = None
    ) -> Dict:
        """
        Train multi-target model for one item.

        Returns:
            Dict with training results and metrics
        """
        start_time = time.time()

        result = {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'success',
            'n_rows': len(df),
            'n_targets': self.config.n_targets,
        }

        try:
            # Get feature columns
            feature_cols = (
                feature_cols_override
                or self.feature_cols_master
                or self.get_feature_columns(df)
            )
            result['n_features'] = len(feature_cols)
            result['feature_cols'] = feature_cols

            # Early validation (Issue #77) - fail fast before expensive operations
            try:
                # Minimum samples needed: warmup rows + enough for train/val/test splits
                min_samples_early = self.config.drop_first_n_rows + 1500
                validation_stats = validate_training_data(
                    df, feature_cols,
                    min_samples=min_samples_early,
                    item_id=item_id
                )
                result['validation_stats'] = validation_stats
            except ValidationError as ve:
                result['status'] = 'error'
                result['error'] = str(ve)
                return result

            if len(feature_cols) == 0:
                result['status'] = 'error'
                result['error'] = 'No feature columns found'
                return result

            if 'low' not in df.columns or 'high' not in df.columns:
                result['status'] = 'error'
                result['error'] = 'Missing required price columns (low/high)'
                return result

            if not set(feature_cols).issubset(df.columns):
                missing = sorted(set(feature_cols) - set(df.columns))
                logger.warning(f"  Missing {len(missing)} feature columns; filling with NaN.")
                feature_df = df.reindex(columns=feature_cols, fill_value=np.nan)
            else:
                feature_df = df[feature_cols]

            # Extract features
            X = feature_df.to_numpy(dtype=np.float32, copy=False)

            # Drop warmup rows
            X = X[self.config.drop_first_n_rows:]
            df_trimmed = df.iloc[self.config.drop_first_n_rows:].reset_index(drop=True)

            # Compute all 108 targets
            y_matrix, target_names = compute_all_targets(df_trimmed, self.config)
            result['target_names'] = target_names
            n = y_matrix.shape[0]
            result['n_samples_after_trim'] = n

            if n < 1000:
                result['status'] = 'error'
                result['error'] = f'Insufficient samples after trimming: {n}'
                return result

            if len(X) < n:
                result['status'] = 'error'
                result['error'] = f'Feature rows shorter than targets: {len(X)} < {n}'
                return result

            # Align features to target length
            X = X[:n]

            # Validate X/y shapes (Issue #77)
            try:
                validate_xy_shapes(X, y_matrix, item_id=item_id)
            except ValidationError as ve:
                result['status'] = 'error'
                result['error'] = str(ve)
                return result

            # Temporal split
            train_end, val_end = self.temporal_split(n)

            X_train = X[:train_end]
            X_val = X[train_end:val_end]
            X_test = X[val_end:]

            y_train = y_matrix[:train_end]
            y_val = y_matrix[train_end:val_end]
            y_test = y_matrix[val_end:]

            # Check for sufficient positive samples in each target
            target_stats = []
            for i, name in enumerate(target_names):
                pos_train = y_train[:, i].sum()
                pos_rate = y_train[:, i].mean()
                target_stats.append({
                    'name': name,
                    'positive_train': int(pos_train),
                    'positive_rate': float(pos_rate),
                    'valid': pos_train >= self.config.min_positive_samples
                })

            result['target_stats'] = target_stats
            valid_targets = sum(1 for ts in target_stats if ts['valid'])
            result['valid_targets'] = valid_targets

            if valid_targets < self.config.n_targets * 0.5:
                result['status'] = 'error'
                result['error'] = f'Too few valid targets: {valid_targets}/{self.config.n_targets}'
                return result

            # Train model
            logger.info(f"  Training MultiLogloss model with {self.config.n_targets} targets...")
            model = self.create_model()

            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=100
            )

            result['iterations_used'] = model.get_best_iteration() or self.config.iterations

            # Evaluate on test set
            logger.info(f"  Evaluating on test set ({len(X_test)} samples)...")
            proba = model.predict_proba(X_test)

            # Extract per-target AUCs
            aucs = []
            auc_values = []
            n_targets = len(target_names)

            # Handle MultiLogloss output format: (N, 2*K) alternating [neg0, pos0, neg1, pos1, ...]
            if proba.shape[1] == 2 * n_targets:
                for i, name in enumerate(target_names):
                    y_prob = proba[:, 2 * i + 1]  # Positive class probability
                    y_true = y_test[:, i]

                    auc = safe_roc_auc(y_true, y_prob)
                    aucs.append({'target': name, 'auc': auc})
                    if auc is not None:
                        auc_values.append(auc)
            else:
                # Direct probability format
                for i, name in enumerate(target_names):
                    y_prob = proba[:, i] if proba.shape[1] > i else np.zeros(len(X_test))
                    y_true = y_test[:, i]

                    auc = safe_roc_auc(y_true, y_prob)
                    aucs.append({'target': name, 'auc': auc})
                    if auc is not None:
                        auc_values.append(auc)

            result['target_aucs'] = aucs
            result['mean_auc'] = float(np.mean(auc_values)) if auc_values else 0.5
            result['targets_scored'] = len(auc_values)
            result['targets_above_52'] = sum(1 for a in auc_values if a > 0.52)
            result['model'] = model

        except Exception as e:
            import traceback
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

        result['train_time_seconds'] = time.time() - start_time

        return result


# =============================================================================
# Model Saving
# =============================================================================

def save_model(
    output_dir: Path,
    run_id: str,
    item_id: int,
    result: Dict
):
    """Save trained model and metadata."""
    item_dir = output_dir / run_id / str(item_id)
    item_dir.mkdir(parents=True, exist_ok=True)

    model = result.get('model')
    if model is None:
        logger.warning(f"No model to save for item {item_id}")
        return

    # Save native CatBoost format (.cbm)
    model_path = item_dir / 'model.cbm'
    model.save_model(str(model_path))
    logger.debug(f"  Saved model to {model_path}")

    # Save metadata
    meta = {
        'item_id': result['item_id'],
        'item_name': result['item_name'],
        'trained_at': datetime.now().isoformat(),
        'n_features': result.get('n_features', 0),
        'n_targets': result.get('n_targets', 0),
        'n_samples': result.get('n_samples_after_trim', 0),
        'mean_auc': result.get('mean_auc', 0.5),
        'targets_scored': result.get('targets_scored', 0),
        'targets_above_52': result.get('targets_above_52', 0),
        'train_time_seconds': result.get('train_time_seconds', 0),
        'iterations_used': result.get('iterations_used', 0),
        'feature_cols': result.get('feature_cols', []),
        'target_names': result.get('target_names', []),
    }

    meta_path = item_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    # Save per-target metrics
    if 'target_aucs' in result:
        aucs_path = item_dir / 'target_aucs.json'
        with open(aucs_path, 'w') as f:
            json.dump(result['target_aucs'], f, indent=2, cls=NumpyEncoder)

    # Remove model from result and free GPU memory
    del result['model']
    gc.collect()


def is_already_trained(output_dir: Path, run_id: str, item_id: int) -> bool:
    """Check if item has already been trained."""
    model_path = output_dir / run_id / str(item_id) / 'model.cbm'
    return model_path.exists()


def iter_with_prefetch(
    items: List[Dict],
    loader: Callable[[Dict], pd.DataFrame],
    prefetch: int
) -> Iterable[Tuple[Dict, pd.DataFrame]]:
    """Prefetch item data to overlap I/O with training."""
    if prefetch <= 0:
        for item in items:
            yield item, loader(item)
        return

    with ThreadPoolExecutor(max_workers=prefetch) as executor:
        item_iter = iter(items)
        queue: deque = deque()

        for _ in range(prefetch):
            try:
                item = next(item_iter)
            except StopIteration:
                break
            queue.append((item, executor.submit(loader, item)))

        while queue:
            item, future = queue.popleft()
            df = future.result()
            yield item, df

            try:
                next_item = next(item_iter)
            except StopIteration:
                next_item = None
            if next_item is not None:
                queue.append((next_item, executor.submit(loader, next_item)))


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RunPod Multi-Target Training with 108 Sequential Fill Targets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_runpod_multitarget.py --run-id 20260110_123456 --all
  python train_runpod_multitarget.py --run-id 20260110_123456 --items 2,10006,10008
  python train_runpod_multitarget.py --run-id 20260110_123456 --all --resume
  python train_runpod_multitarget.py --run-id 20260110_123456 --items 2 --cpu
        """
    )

    # Item selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Train all items')
    group.add_argument('--items', type=str, help='Comma-separated item IDs')

    # Required
    parser.add_argument('--run-id', required=True, help='GCS run ID from prepare_training_data.py')

    # Options
    parser.add_argument('--bucket', default=DEFAULT_BUCKET, help=f'GCS bucket (default: {DEFAULT_BUCKET})')
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR, help=f'Local cache (default: {DEFAULT_CACHE_DIR})')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help=f'Output dir (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--resume', action='store_true', help='Skip already-trained items')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--local', action='store_true', help='Use local data only (from prepare_runpod_data.py)')
    parser.add_argument('--limit', type=int, help='Limit number of items (for testing)')
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 4, help='CPU threads to use')
    parser.add_argument('--numba-threads', type=int, help='Numba threads (default: --threads)')
    parser.add_argument('--prefetch', type=int, default=2, help='Prefetch items to overlap I/O')
    parser.add_argument('--gc-every', type=int, default=10, help='Run gc.collect every N items')
    parser.add_argument('--periods-per-hour', type=int, default=12, help='Data periods per hour')
    parser.add_argument('--no-numba', action='store_true', help='Disable Numba acceleration')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--json-logs', action='store_true', help='Output structured JSON logs (machine-parseable)')
    parser.add_argument('--log-every', type=int, default=1, help='Log progress every N items (default: every item)')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    global USE_NUMBA
    USE_NUMBA = HAS_NUMBA and not args.no_numba
    configure_threading(args.threads, args.numba_threads)
    if not HAS_NUMBA:
        logger.warning("Numba not available; falling back to NumPy targets.")

    # Print banner
    logger.info("=" * 70)
    logger.info("GePT RunPod Multi-Target Training")
    logger.info("108 Sequential Fill Targets (18 hours × 6 offsets)")
    logger.info("=" * 70)

    # Validate GCS credentials early if using GCS (not --local mode)
    if not args.local and HAS_GCS:
        logger.info("\nValidating GCS credentials...")
        validate_gcs_credentials()

    # Create config
    config = TrainingConfig(
        bucket_name=args.bucket,
        use_gpu=not args.cpu,
        cpu_threads=args.threads,
        numba_threads=args.numba_threads,
        periods_per_hour=args.periods_per_hour
    )

    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"GPU: {config.use_gpu}")
    logger.info(f"CPU threads: {config.cpu_threads}")
    logger.info(f"Numba enabled: {USE_NUMBA}")
    if USE_NUMBA:
        logger.info(f"Numba threads: {config.numba_threads or config.cpu_threads}")
    logger.info(f"Time windows: {config.hours}")
    logger.info(f"Periods per hour: {config.periods_per_hour}")
    logger.info(f"Offsets: {[f'{o*100:.2f}%' for o in config.offsets]}")
    logger.info(f"Total targets per item: {config.n_targets}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Local data only: {args.local}")
    logger.info(f"Prefetch: {args.prefetch}")
    logger.info(f"GC every: {args.gc_every} items")

    # Create directories
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config (from local or GCS)
    source = "local" if args.local else "GCS"
    logger.info(f"\nLoading configuration from {source}...")
    gcs_bucket = None
    if not args.local and HAS_GCS:
        gcs_client = storage.Client()
        gcs_bucket = gcs_client.bucket(args.bucket)

    gcs_config = load_config(
        args.bucket,
        args.run_id,
        args.cache_dir,
        use_local=args.local,
        gcs_bucket=gcs_bucket
    )

    all_items = gcs_config['items']
    logger.info(f"Total items available: {len(all_items)}")

    # Filter items
    if args.items:
        item_ids = set(int(x.strip()) for x in args.items.split(','))
        items = [i for i in all_items if i['item_id'] in item_ids]
    else:
        items = all_items

    if args.limit:
        items = items[:args.limit]

    logger.info(f"Items to train: {len(items)}")

    # Apply resume logic
    if args.resume:
        original_count = len(items)
        items = [
            i for i in items
            if not is_already_trained(args.output_dir, args.run_id, i['item_id'])
        ]
        skipped = original_count - len(items)
        if skipped > 0:
            logger.info(f"Skipped {skipped} already-trained items (--resume)")

    if not items:
        logger.info("No items to train. Exiting.")
        return

    # Create trainer
    trainer = MultiTargetTrainer(config)

    feature_cols_master = gcs_config.get('feature_cols')
    if isinstance(feature_cols_master, list) and feature_cols_master:
        feature_cols_master = [
            c for c in feature_cols_master
            if c not in {'low', 'high', 'avg_high_price', 'avg_low_price'}
        ]
        if feature_cols_master:
            trainer.set_feature_columns(feature_cols_master)
            logger.info(f"Using {len(feature_cols_master)} feature columns from config.")
        else:
            feature_cols_master = None
            logger.warning("Config feature_cols became empty after filtering; inferring from data.")
    else:
        feature_cols_master = None
        logger.info("Feature columns not found in config; inferring from data.")

    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    start_time = time.time()
    results = []
    errors = []
    total_items = len(items)
    processed = 0

    # Initialize training metrics (Issue #75)
    training_metrics = TrainingMetrics(
        total_items=total_items,
        run_id=args.run_id,
        log_json=args.json_logs,
        log_every_n=args.log_every,
    )

    # Log training start with config
    training_metrics.log_training_start({
        'hours': config.hours,
        'offsets': config.offsets,
        'n_targets': config.n_targets,
        'iterations': config.iterations,
        'depth': config.depth,
        'learning_rate': config.learning_rate,
        'use_gpu': config.use_gpu,
        'cpu_threads': config.cpu_threads,
        'numba_enabled': USE_NUMBA,
    })

    pbar = tqdm(total=total_items, desc="Training") if HAS_TQDM else None

    def process_item(item_config: Dict, df: pd.DataFrame, idx: int) -> None:
        nonlocal processed
        item_id = item_config['item_id']
        item_name = item_config.get('item_name', str(item_id))

        if not HAS_TQDM and not args.json_logs:
            logger.info(f"\n[{idx}/{total_items}] Training item {item_id} ({item_name})...")

        # Use structured telemetry timer (Issue #75)
        with training_metrics.item_timer(item_id, item_name) as timer:
            try:
                result = trainer.train_item(item_id, item_name, df)

                if result['status'] == 'success':
                    save_model(args.output_dir, args.run_id, item_id, result)

                    # Update telemetry with model metrics
                    timer.set_auc(result['mean_auc'])
                    timer.set_targets_scored(
                        result.get('targets_scored', 0),
                        result.get('targets_above_52', 0)
                    )
                    timer.set_model_info(
                        n_samples=result.get('n_samples_after_trim'),
                        n_features=result.get('n_features'),
                        iterations_used=result.get('iterations_used'),
                    )
                    timer.set_status('success')

                    if not HAS_TQDM and not args.json_logs:
                        logger.info(
                            f"  Mean AUC: {result['mean_auc']:.4f}, "
                            f"Targets >0.52: {result['targets_above_52']}/{config.n_targets}, "
                            f"Scored: {result.get('targets_scored', 0)}/{config.n_targets}, "
                            f"Time: {result['train_time_seconds']:.1f}s"
                        )

                    results.append({
                        'item_id': item_id,
                        'item_name': item_name,
                        'status': 'success',
                        'mean_auc': result['mean_auc'],
                        'targets_above_52': result['targets_above_52'],
                        'targets_scored': result.get('targets_scored', 0),
                        'train_time_seconds': result['train_time_seconds'],
                    })
                else:
                    timer.set_status('error', result.get('error', 'unknown'))
                    if not HAS_TQDM and not args.json_logs:
                        logger.error(f"  Error: {result.get('error', 'unknown')}")
                    errors.append(result)

            except Exception as e:
                import traceback
                error_result = {
                    'item_id': item_id,
                    'item_name': item_name,
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                errors.append(error_result)
                timer.set_status('error', str(e))
                if not HAS_TQDM and not args.json_logs:
                    logger.error(f"  Exception: {e}")

        processed += 1
        if pbar:
            pbar.update(1)

        if args.gc_every and processed % args.gc_every == 0:
            gc.collect()

        # Log progress periodically (legacy format when not using JSON)
        if not HAS_TQDM and not args.json_logs and processed % 25 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed * 60
            remaining = (total_items - processed) / (rate / 60) if rate > 0 else 0
            logger.info(
                f"\n=== Progress: {processed}/{total_items}, "
                f"{rate:.1f} items/min, ~{remaining/60:.1f}h remaining ===\n"
            )

    def load_for_item(item_config: Dict, columns: Optional[List[str]]) -> pd.DataFrame:
        item_id = item_config['item_id']
        return load_item_data(
            args.bucket,
            args.run_id,
            item_id,
            args.cache_dir,
            use_local=args.local,
            gcs_bucket=gcs_bucket,
            columns=columns
        )

    remaining_items = list(items)

    if feature_cols_master is None:
        while remaining_items and feature_cols_master is None:
            item_config = remaining_items.pop(0)
            try:
                df = load_for_item(item_config, columns=None)
                inferred = trainer.get_feature_columns(df)
                if not inferred:
                    errors.append({
                        'item_id': item_config['item_id'],
                        'item_name': item_config.get('item_name', str(item_config['item_id'])),
                        'status': 'error',
                        'error': 'No feature columns found to infer schema'
                    })
                    processed += 1
                    if pbar:
                        pbar.update(1)
                    continue

                feature_cols_master = inferred
                trainer.set_feature_columns(feature_cols_master)
                logger.info(f"Inferred {len(feature_cols_master)} feature columns from item {item_config['item_id']}.")
                process_item(item_config, df, processed + 1)
            except Exception as e:
                import traceback
                errors.append({
                    'item_id': item_config['item_id'],
                    'item_name': item_config.get('item_name', str(item_config['item_id'])),
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                processed += 1
                if pbar:
                    pbar.update(1)

    if remaining_items:
        if feature_cols_master:
            columns = sorted(set(feature_cols_master + ['low', 'high']))
        else:
            columns = None

        def loader(item_config: Dict) -> pd.DataFrame:
            return load_for_item(item_config, columns=columns)

        for idx, (item_config, df) in enumerate(
            iter_with_prefetch(remaining_items, loader, args.prefetch),
            start=processed + 1
        ):
            process_item(item_config, df, idx)

    if pbar:
        pbar.close()

    # Log structured training summary (Issue #75)
    telemetry_summary = training_metrics.log_summary()

    # Summary
    total_time = time.time() - start_time
    successful = len(results)

    if not args.json_logs:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Successful: {successful}/{len(items)}")
        logger.info(f"Errors: {len(errors)}")
        logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

        if successful > 0:
            avg_auc = np.mean([r['mean_auc'] for r in results])
            avg_time = np.mean([r['train_time_seconds'] for r in results])
            logger.info(f"Average AUC: {avg_auc:.4f}")
            logger.info(f"Average time per item: {avg_time:.1f}s")
            logger.info(f"Rate: {successful / total_time * 60:.1f} items/minute")

        logger.info(f"Models saved to: {args.output_dir / args.run_id}")

    # Save training summary
    summary = {
        'run_id': args.run_id,
        'completed_at': datetime.now().isoformat(),
        'config': {
            'hours': config.hours,
            'offsets': config.offsets,
            'n_targets': config.n_targets,
            'periods_per_hour': config.periods_per_hour,
            'iterations': config.iterations,
            'depth': config.depth,
            'learning_rate': config.learning_rate,
            'use_gpu': config.use_gpu,
            'cpu_threads': config.cpu_threads,
            'numba_threads': config.numba_threads,
            'gpu_ram_part': config.gpu_ram_part,
            'od_wait': config.od_wait,
            'use_numba': USE_NUMBA,
            'prefetch': args.prefetch,
            'gc_every': args.gc_every,
        },
        'total_items': len(items),
        'successful': successful,
        'errors': len(errors),
        'total_time_seconds': total_time,
        'items_per_minute': successful / total_time * 60 if total_time > 0 else 0,
        'average_auc': float(np.mean([r['mean_auc'] for r in results])) if results else None,
        'average_time_per_item': float(np.mean([r['train_time_seconds'] for r in results])) if results else None,
        'results': results,
        # Include telemetry data (Issue #75)
        'telemetry': telemetry_summary,
        'detailed_item_metrics': training_metrics.get_completed_items(),
    }

    summary_path = args.output_dir / args.run_id / 'training_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Save errors if any
    if errors:
        errors_path = args.output_dir / args.run_id / 'errors.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2, cls=NumpyEncoder)
        logger.warning(f"Errors saved to: {errors_path}")

    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
