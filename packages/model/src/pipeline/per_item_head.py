"""
Per-Item Head Architecture
==========================

Small specialized models that sit on top of the shared PatchTST base.
Each item gets its own tiny MLP that learns item-specific patterns.

Architecture:
    Shared PatchTST Base (frozen) → Per-Item MLP → Quantile Predictions

Benefits:
- Item-specific scaling and bias
- Can learn price floor/ceiling awareness (alch prices)
- Different volatility patterns per item
- Much faster to train than full model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig


class PerItemHead(nn.Module):
    """
    Small MLP head specialized for a single item.

    Takes the pooled transformer output and produces item-specific
    quantile predictions. Much smaller than the shared heads.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 64,
        n_horizons: int = 7,
        n_quantiles: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

        # Small MLP for this specific item
        # Learns item-specific transformations
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate output projections for high and low
        self.high_proj = nn.Linear(hidden_dim, n_horizons * n_quantiles)
        self.low_proj = nn.Linear(hidden_dim, n_horizons * n_quantiles)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pooled: Pooled transformer output (batch, d_model)

        Returns:
            Dict with 'high_quantiles' and 'low_quantiles'
            each of shape (batch, n_horizons, n_quantiles)
        """
        x = self.mlp(pooled)

        high = self.high_proj(x)
        low = self.low_proj(x)

        # Reshape to (batch, n_horizons, n_quantiles)
        batch_size = pooled.shape[0]
        high = high.view(batch_size, self.n_horizons, self.n_quantiles)
        low = low.view(batch_size, self.n_horizons, self.n_quantiles)

        return {
            'high_quantiles': high,
            'low_quantiles': low
        }


class PatchTSTWithItemHead(nn.Module):
    """
    PatchTST base model with a specialized per-item head.

    The base model is frozen - only the per-item head is trained.
    This allows fast fine-tuning for specific items.
    """

    def __init__(
        self,
        base_model: PatchTSTModel,
        item_head: PerItemHead,
        freeze_base: bool = True
    ):
        super().__init__()
        self.base = base_model
        self.item_head = item_head

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def get_base_features(
        self,
        recent: torch.Tensor,
        medium: torch.Tensor,
        long: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract pooled features from base model without prediction heads.
        """
        c = self.base.config

        # Encode each resolution
        recent_patches = self.base.recent_encoder(recent)
        medium_patches = self.base.medium_encoder(medium)
        long_patches = self.base.long_encoder(long)

        # Concatenate all patches
        all_patches = torch.cat(
            [recent_patches, medium_patches, long_patches], dim=1
        )

        # Add item token if provided
        if item_ids is not None:
            item_embed = self.base.item_embedding(item_ids)
            item_token = self.base.item_proj(item_embed).unsqueeze(1)
            all_patches = torch.cat([item_token, all_patches], dim=1)

        # Apply positional encoding and transformer
        x = self.base.pos_encoder(all_patches)
        x = self.base.transformer(x)

        # Mean pooling and normalization
        x = x.mean(dim=1)
        x = self.base.pool_norm(x)

        return x

    def forward(
        self,
        recent: torch.Tensor,
        medium: torch.Tensor,
        long: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: base features → per-item head → predictions
        """
        # Get pooled features from frozen base
        with torch.no_grad() if not any(p.requires_grad for p in self.base.parameters()) else torch.enable_grad():
            pooled = self.get_base_features(recent, medium, long, item_ids)

        # Per-item head predictions
        return self.item_head(pooled)


def extract_item_samples(
    data: Dict[str, np.ndarray],
    item_idx: int
) -> Dict[str, np.ndarray]:
    """
    Extract samples for a specific item from a chunk.
    """
    mask = data['item_ids'] == item_idx
    return {
        'recent': data['recent'][mask],
        'medium': data['medium'][mask],
        'long': data['long'][mask],
        'item_ids': data['item_ids'][mask],
        'targets': data['targets'][mask],
    }


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    target = target.expand_as(pred)

    errors = target - pred
    losses = []
    for i, q in enumerate(quantiles):
        error = errors[..., i]
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss)

    return torch.mean(torch.stack(losses))


def combined_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
) -> torch.Tensor:
    """Combined loss for high and low predictions."""
    high_target = targets[..., 0]
    low_target = targets[..., 1]

    high_loss = quantile_loss(outputs['high_quantiles'], high_target, quantiles)
    low_loss = quantile_loss(outputs['low_quantiles'], low_target, quantiles)

    return high_loss + low_loss
