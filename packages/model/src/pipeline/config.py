"""
Configuration dataclasses for the training pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import yaml


@dataclass
class DataConfig:
    """Configuration for data extraction and features."""
    recent_hours: int = 24
    medium_days: int = 7
    long_days: int = 30
    horizons: Tuple[int, ...] = (1, 2, 4, 8, 12, 24, 48)
    max_gap_pct: float = 0.05
    outlier_threshold: float = 0.5

    @property
    def recent_len(self) -> int:
        return self.recent_hours * 12

    @property
    def medium_len(self) -> int:
        return self.medium_days * 24

    @property
    def long_len(self) -> int:
        return self.long_days * 6

    @property
    def min_history_days(self) -> int:
        return self.long_days


@dataclass
class ModelConfig:
    """Configuration for PatchTST model architecture."""
    recent_len: int = 288
    recent_features: int = 6
    medium_len: int = 168
    medium_features: int = 10
    long_len: int = 180
    long_features: int = 10
    patch_size: int = 16
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 768
    dropout: float = 0.1
    n_horizons: int = 7
    n_quantiles: int = 5
    quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_items: int = 500
    item_embed_dim: int = 32
    # Volume head settings
    enable_volume_head: bool = True
    volume_quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    volume_hidden_dim: int = 64


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    num_workers: int = 4
    checkpoint_every: int = 5


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path("data")
    model_dir: Path = Path("models/patchtst_v1")

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            output_dir=Path(data.get("output_dir", "data")),
            model_dir=Path(data.get("model_dir", "models/patchtst_v1")),
        )
