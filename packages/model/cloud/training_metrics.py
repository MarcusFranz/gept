"""
Training Pipeline Telemetry - Structured Progress Logging

Provides machine-parseable JSON logging for training pipeline monitoring:
- Per-item timing metrics
- Memory usage tracking
- GPU utilization (if available)
- Progress estimation with variance
- Event-based structured logging

Usage:
    from training_metrics import TrainingMetrics

    metrics = TrainingMetrics(total_items=400, run_id="20260114_123456")

    for item_id, item_name in items:
        with metrics.item_timer(item_id, item_name) as timer:
            # ... train model ...
            timer.set_auc(0.72)
            timer.set_status("success")

    metrics.log_summary()
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> Optional[float]:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
    except Exception:
        return None


def get_gpu_memory_mb() -> Optional[Dict[str, float]]:
    """Get GPU memory usage in MB (NVIDIA only)."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                if len(parts) >= 3:
                    return {
                        'used_mb': float(parts[0].strip()),
                        'total_mb': float(parts[1].strip()),
                        'utilization_percent': float(parts[2].strip()),
                    }
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass
    return None


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


@dataclass
class ItemMetrics:
    """Metrics for a single item training run."""
    item_id: int
    item_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "in_progress"
    error: Optional[str] = None

    # Model metrics
    mean_auc: Optional[float] = None
    targets_scored: Optional[int] = None
    targets_above_52: Optional[int] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    iterations_used: Optional[int] = None

    # Resource metrics
    memory_mb_start: Optional[float] = None
    memory_mb_end: Optional[float] = None
    gpu_memory_start: Optional[Dict[str, float]] = None
    gpu_memory_end: Optional[Dict[str, float]] = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'item_name': self.item_name,
            'status': self.status,
            'duration_seconds': round(self.duration_seconds, 2),
            'mean_auc': self.mean_auc,
            'targets_scored': self.targets_scored,
            'targets_above_52': self.targets_above_52,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'iterations_used': self.iterations_used,
            'memory_mb_start': self.memory_mb_start,
            'memory_mb_end': self.memory_mb_end,
            'gpu_memory_start': self.gpu_memory_start,
            'gpu_memory_end': self.gpu_memory_end,
            'error': self.error,
        }


class ItemTimer:
    """Context manager for timing individual item training."""

    def __init__(self, metrics: 'TrainingMetrics', item_id: int, item_name: str):
        self._metrics = metrics
        self._item = ItemMetrics(
            item_id=item_id,
            item_name=item_name,
            memory_mb_start=get_memory_usage_mb(),
            gpu_memory_start=get_gpu_memory_mb(),
        )

    def set_auc(self, auc: float) -> None:
        self._item.mean_auc = auc

    def set_targets_scored(self, scored: int, above_52: int) -> None:
        self._item.targets_scored = scored
        self._item.targets_above_52 = above_52

    def set_model_info(
        self,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        iterations_used: Optional[int] = None
    ) -> None:
        if n_samples is not None:
            self._item.n_samples = n_samples
        if n_features is not None:
            self._item.n_features = n_features
        if iterations_used is not None:
            self._item.iterations_used = iterations_used

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        self._item.status = status
        self._item.error = error

    def __enter__(self) -> 'ItemTimer':
        self._metrics._log_item_start(self._item)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._item.end_time = time.time()
        self._item.memory_mb_end = get_memory_usage_mb()
        self._item.gpu_memory_end = get_gpu_memory_mb()

        if exc_type is not None:
            self._item.status = "error"
            self._item.error = str(exc_val)

        self._metrics._log_item_complete(self._item)
        return False  # Don't suppress exceptions


class TrainingMetrics:
    """
    Structured telemetry for the training pipeline.

    Provides JSON-formatted logging for:
    - Training start/end events
    - Per-item progress with timing and resource usage
    - ETA estimation with variance
    - Summary statistics
    """

    def __init__(
        self,
        total_items: int,
        run_id: str,
        log_json: bool = True,
        log_every_n: int = 1,
    ):
        """
        Initialize training metrics.

        Args:
            total_items: Total number of items to train
            run_id: Training run identifier
            log_json: If True, log events as JSON (machine-parseable)
            log_every_n: Log progress every N items (default 1 = every item)
        """
        self.total_items = total_items
        self.run_id = run_id
        self.log_json = log_json
        self.log_every_n = log_every_n

        self.start_time = time.time()
        self.items_completed = 0
        self.items_successful = 0
        self.items_failed = 0

        self._item_durations: List[float] = []
        self._item_aucs: List[float] = []
        self._completed_items: List[Dict[str, Any]] = []

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event."""
        event = {
            'event': event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'run_id': self.run_id,
            **data
        }

        if self.log_json:
            logger.info(json.dumps(event, cls=NumpyEncoder))
        else:
            # Human-readable format
            msg_parts = [f"[{event_type}]"]
            for key, value in data.items():
                if value is not None:
                    msg_parts.append(f"{key}={value}")
            logger.info(" ".join(msg_parts))

    def log_training_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Log training start event."""
        data = {
            'total_items': self.total_items,
            'start_memory_mb': get_memory_usage_mb(),
            'gpu_info': get_gpu_memory_mb(),
        }
        if config:
            data['config'] = config

        self._log_event('training_start', data)

    def _log_item_start(self, item: ItemMetrics) -> None:
        """Log item training start (internal)."""
        if self.log_every_n <= 1 or self.items_completed % self.log_every_n == 0:
            self._log_event('item_start', {
                'item_id': item.item_id,
                'item_name': item.item_name,
                'progress': f"{self.items_completed + 1}/{self.total_items}",
                'memory_mb': item.memory_mb_start,
                'gpu_memory': item.gpu_memory_start,
            })

    def _log_item_complete(self, item: ItemMetrics) -> None:
        """Log item training completion (internal)."""
        self.items_completed += 1

        if item.status == "success":
            self.items_successful += 1
            self._item_durations.append(item.duration_seconds)
            if item.mean_auc is not None:
                self._item_aucs.append(item.mean_auc)
        else:
            self.items_failed += 1

        self._completed_items.append(item.to_dict())

        # Calculate ETA with variance
        eta_info = self._calculate_eta()

        data = {
            'item_id': item.item_id,
            'item_name': item.item_name,
            'status': item.status,
            'duration_seconds': round(item.duration_seconds, 2),
            'mean_auc': item.mean_auc,
            'targets_scored': item.targets_scored,
            'targets_above_52': item.targets_above_52,
            'progress': {
                'completed': self.items_completed,
                'total': self.total_items,
                'percent': round(100 * self.items_completed / self.total_items, 1),
                'successful': self.items_successful,
                'failed': self.items_failed,
            },
            'memory_mb': item.memory_mb_end,
            'gpu_memory': item.gpu_memory_end,
            'eta': eta_info,
        }

        if item.error:
            data['error'] = item.error

        if self.log_every_n <= 1 or self.items_completed % self.log_every_n == 0:
            self._log_event('item_complete', data)

    def _calculate_eta(self) -> Dict[str, Any]:
        """Calculate ETA with variance estimation."""
        if not self._item_durations:
            return {'eta_seconds': None, 'eta_variance': None}

        remaining = self.total_items - self.items_completed
        avg_duration = mean(self._item_durations)
        eta_seconds = remaining * avg_duration

        eta_info = {
            'eta_seconds': round(eta_seconds, 0),
            'eta_minutes': round(eta_seconds / 60, 1),
            'eta_hours': round(eta_seconds / 3600, 2),
            'avg_duration_seconds': round(avg_duration, 2),
            'items_remaining': remaining,
        }

        # Add variance if we have enough samples
        if len(self._item_durations) >= 3:
            duration_stdev = stdev(self._item_durations)
            eta_info['duration_stdev'] = round(duration_stdev, 2)
            # 95% confidence interval for remaining time
            eta_variance = remaining * duration_stdev * 1.96
            eta_info['eta_variance_seconds'] = round(eta_variance, 0)
            eta_info['eta_range_minutes'] = [
                round((eta_seconds - eta_variance) / 60, 1),
                round((eta_seconds + eta_variance) / 60, 1),
            ]

        return eta_info

    def item_timer(self, item_id: int, item_name: str) -> ItemTimer:
        """Create a context manager for timing an item."""
        return ItemTimer(self, item_id, item_name)

    def log_progress(self) -> None:
        """Log current progress (call periodically)."""
        elapsed = time.time() - self.start_time
        rate = self.items_completed / elapsed * 60 if elapsed > 0 else 0
        eta_info = self._calculate_eta()

        self._log_event('progress', {
            'completed': self.items_completed,
            'total': self.total_items,
            'successful': self.items_successful,
            'failed': self.items_failed,
            'elapsed_seconds': round(elapsed, 0),
            'elapsed_minutes': round(elapsed / 60, 1),
            'rate_items_per_minute': round(rate, 2),
            'memory_mb': get_memory_usage_mb(),
            'gpu_memory': get_gpu_memory_mb(),
            'eta': eta_info,
        })

    def log_summary(self) -> Dict[str, Any]:
        """Log final training summary and return summary dict."""
        elapsed = time.time() - self.start_time

        summary = {
            'run_id': self.run_id,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'total_items': self.total_items,
            'items_completed': self.items_completed,
            'items_successful': self.items_successful,
            'items_failed': self.items_failed,
            'success_rate': round(100 * self.items_successful / max(1, self.items_completed), 1),
            'total_time_seconds': round(elapsed, 1),
            'total_time_minutes': round(elapsed / 60, 1),
            'total_time_hours': round(elapsed / 3600, 2),
            'rate_items_per_minute': round(self.items_completed / elapsed * 60, 2) if elapsed > 0 else 0,
            'memory_mb_final': get_memory_usage_mb(),
            'gpu_memory_final': get_gpu_memory_mb(),
        }

        # Add timing statistics
        if self._item_durations:
            summary['timing'] = {
                'avg_seconds': round(mean(self._item_durations), 2),
                'min_seconds': round(min(self._item_durations), 2),
                'max_seconds': round(max(self._item_durations), 2),
            }
            if len(self._item_durations) >= 3:
                summary['timing']['stdev_seconds'] = round(stdev(self._item_durations), 2)

        # Add AUC statistics
        if self._item_aucs:
            summary['auc'] = {
                'mean': round(mean(self._item_aucs), 4),
                'min': round(min(self._item_aucs), 4),
                'max': round(max(self._item_aucs), 4),
            }
            if len(self._item_aucs) >= 3:
                summary['auc']['stdev'] = round(stdev(self._item_aucs), 4)

        self._log_event('training_complete', summary)

        return summary

    def get_completed_items(self) -> List[Dict[str, Any]]:
        """Get list of all completed item metrics."""
        return self._completed_items
