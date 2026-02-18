from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import GePT3ModelConfig
from .model import GePT3UnifiedModel


@dataclass(frozen=True)
class GePT3Artifact:
    """Serialized GePT3 artifact (model + schema + normalization)."""

    model_config: GePT3ModelConfig
    feature_cols: list[str]
    horizons_hours: list[int]
    offsets: list[float]
    feature_mean: list[float]
    feature_std: list[float]
    model_state_dict: dict[str, Any]
    notes: dict[str, Any] | None = None

    def save_dir(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.model_state_dict, out / "model.pt")
        meta = {
            "model_config": asdict(self.model_config),
            "feature_cols": self.feature_cols,
            "horizons_hours": self.horizons_hours,
            "offsets": self.offsets,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "notes": self.notes or {},
        }
        (out / "artifact.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return out

    @classmethod
    def load_dir(cls, in_dir: str | Path) -> "GePT3Artifact":
        src = Path(in_dir)
        meta = json.loads((src / "artifact.json").read_text())
        state = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
        return cls(
            model_config=GePT3ModelConfig(**meta["model_config"]),
            feature_cols=[str(v) for v in meta["feature_cols"]],
            horizons_hours=[int(v) for v in meta["horizons_hours"]],
            offsets=[float(v) for v in meta["offsets"]],
            feature_mean=[float(v) for v in meta["feature_mean"]],
            feature_std=[float(v) for v in meta["feature_std"]],
            model_state_dict=state,
            notes=dict(meta.get("notes", {})),
        )

    def build_model(self) -> GePT3UnifiedModel:
        m = GePT3UnifiedModel(self.model_config)
        m.load_state_dict(self.model_state_dict)
        m.eval()
        return m

    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.asarray(self.feature_std, dtype=np.float32)
        std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        return (x - mean) / std
