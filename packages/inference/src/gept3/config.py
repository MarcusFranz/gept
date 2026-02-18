from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GePT3ModelConfig:
    """Unified GePT3 model configuration."""

    n_items: int
    n_features: int
    n_horizons: int
    n_offsets: int

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: float = 4.0
    dropout: float = 0.10

    # Hybrid encoder settings (per-item temporal sequence encoder).
    temporal_len: int = 0
    temporal_n_heads: int = 4
    temporal_n_layers: int = 2

    # Number of quantiles for each side (high/low) in the price head.
    n_price_quantiles: int = 5

    # Enable/disable heads. V3 training starts with sell head active.
    enable_sell_head: bool = True
    enable_buy_head: bool = True
    enable_price_head: bool = True
    enable_regime_head: bool = True

    # Keep for metadata only (output slots for price head ordering).
    price_quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.n_horizons <= 0:
            raise ValueError("n_horizons must be > 0")
        if self.n_offsets <= 0:
            raise ValueError("n_offsets must be > 0")
        if self.n_price_quantiles <= 0:
            raise ValueError("n_price_quantiles must be > 0")
        if self.n_items <= 0:
            raise ValueError("n_items must be > 0")
        if self.n_features <= 0:
            raise ValueError("n_features must be > 0")
        if self.temporal_len < 0:
            raise ValueError("temporal_len must be >= 0")
        if self.temporal_n_heads <= 0:
            raise ValueError("temporal_n_heads must be > 0")
        if self.d_model % self.temporal_n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by temporal_n_heads ({self.temporal_n_heads})"
            )
        if self.temporal_n_layers <= 0:
            raise ValueError("temporal_n_layers must be > 0")
        if len(tuple(self.price_quantiles)) != self.n_price_quantiles:
            raise ValueError(
                f"price_quantiles length ({len(tuple(self.price_quantiles))}) must match "
                f"n_price_quantiles ({self.n_price_quantiles})"
            )
