from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .config import FillProbConfig
from .model import SellFillMLP, SellFillModelConfig


@dataclass(frozen=True)
class SellFillArtifact:
    """Serialized artifact for inference (model + feature schema + normalization)."""

    model_config: SellFillModelConfig
    fillprob_config: FillProbConfig
    feature_cols: List[str]
    feature_mean: List[float]
    feature_std: List[float]
    # How to map OSRS item_id -> embedding index. V1 uses identity mapping:
    #   item_idx = clamp(item_id, 0, n_items-1)
    model_state_dict: Dict[str, Any]
    item_index_mode: str = "identity"

    def save_dir(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.model_state_dict, out / "model.pt")

        meta = {
            "model_config": asdict(self.model_config),
            "fillprob_config": asdict(self.fillprob_config),
            "feature_cols": self.feature_cols,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "item_index_mode": self.item_index_mode,
        }
        (out / "artifact.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return out

    @classmethod
    def load_dir(cls, in_dir: str | Path) -> "SellFillArtifact":
        p = Path(in_dir)
        meta = json.loads((p / "artifact.json").read_text())
        state = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
        return cls(
            model_config=SellFillModelConfig(**meta["model_config"]),
            fillprob_config=FillProbConfig(**meta["fillprob_config"]),
            feature_cols=list(meta["feature_cols"]),
            feature_mean=list(meta["feature_mean"]),
            feature_std=list(meta["feature_std"]),
            item_index_mode=str(meta.get("item_index_mode", "identity")),
            model_state_dict=state,
        )

    def build_model(self) -> SellFillMLP:
        m = SellFillMLP(self.model_config)
        m.load_state_dict(self.model_state_dict)
        m.eval()
        return m

    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.asarray(self.feature_std, dtype=np.float32)
        std = np.where(std > 1e-8, std, 1.0)
        return (x - mean) / std
