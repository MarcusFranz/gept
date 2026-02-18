from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SellFillModelConfig:
    n_items: int
    n_features: int
    item_embed_dim: int = 16
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.10


class SellFillMLP(nn.Module):
    """Predict P(sell fills within horizon) for a given (features, item_id, offset, horizon).

    This is intentionally simple so we can train fast and run inference on Ampere CPU.
    """

    def __init__(self, config: SellFillModelConfig):
        super().__init__()
        self.config = config

        self.item_embedding = nn.Embedding(config.n_items, config.item_embed_dim)

        in_dim = config.n_features + config.item_embed_dim + 2  # + offset, + horizon
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(config.n_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, 1))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,              # (batch, n_features)
        item_idx: torch.Tensor,       # (batch,)
        offset: torch.Tensor,         # (batch, 1)
        horizon_hours: torch.Tensor,  # (batch, 1) normalized or raw
    ) -> torch.Tensor:
        # CoreML conversion prefers int32 inputs; embedding needs int64.
        embed = self.item_embedding(item_idx.long())
        z = torch.cat([x, embed, offset, horizon_hours], dim=1)
        return self.net(z).squeeze(1)  # logits (batch,)

