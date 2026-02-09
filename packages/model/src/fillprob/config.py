from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class FillProbConfig:
    """Configuration shared across dataset building, training, and inference."""

    # Hours ahead for "fills within horizon" targets. Must be <= 24 for V3.
    # Default to full 1..24h to match V3's "remaining_hours" semantics.
    horizons_hours: Sequence[int] = tuple(range(1, 25))

    # Offsets relative to current_high for sells:
    # sell_price = current_high * (1 + offset)
    # Include some negative offsets so the model learns "below market" sells.
    offsets: Sequence[float] = (-0.02, -0.01, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)

    # Target data granularity. Training uses 5m.
    granularity: str = "5m"

    # Optional tax rate reference (not directly used for sell-fill labels).
    tax_rate: float = 0.02

    def __post_init__(self):
        # Guardrails for V3: no 48h.
        if any(h > 24 for h in self.horizons_hours):
            raise ValueError(f"horizons_hours must be <= 24, got: {self.horizons_hours}")

        if self.granularity not in ("5m", "1m"):
            raise ValueError(f"Unsupported granularity: {self.granularity}")

    @property
    def periods_per_hour(self) -> int:
        if self.granularity == "5m":
            return 12
        if self.granularity == "1m":
            return 60
        raise ValueError(f"Unsupported granularity: {self.granularity}")

    @property
    def horizon_periods(self) -> dict[int, int]:
        pph = self.periods_per_hour
        return {h: h * pph for h in self.horizons_hours}
