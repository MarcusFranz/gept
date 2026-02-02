"""
PatchTST Model Architecture
===========================

Multi-resolution Patch Time Series Transformer for price prediction.

Three encoders process different temporal resolutions:
- Recent: 5-min, 24h lookback, 288 timesteps, 6 features
- Medium: 1-hour, 7-day lookback, 168 timesteps, 10 features
- Long: 4-hour, 30-day lookback, 180 timesteps, 10 features

The model predicts quantiles (p10, p30, p50, p70, p90) of future
price movements at 7 horizons (1h, 2h, 4h, 8h, 12h, 24h, 48h).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import ModelConfig


class PatchEmbedding(nn.Module):
    """
    Project time series patches to model dimension.

    Takes a sequence of shape (batch, seq_len, n_features) and produces
    patch embeddings of shape (batch, n_patches, d_model).
    """

    def __init__(self, patch_size: int, n_features: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)

        Returns:
            Patch embeddings of shape (batch, n_patches, d_model)
        """
        batch_size, seq_len, n_features = x.shape

        # Pad sequence to be divisible by patch_size
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        n_patches = (seq_len + pad_len) // self.patch_size

        # Reshape to (batch, n_patches, patch_size * n_features)
        x = x.reshape(batch_size, n_patches, self.patch_size * n_features)

        return self.projection(x)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.

    Uses the standard sin/cos positional encoding from "Attention is All You Need".
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Position-encoded tensor of same shape
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResolutionEncoder(nn.Module):
    """
    Encode a single temporal resolution with learnable resolution embedding.

    Each resolution (recent/medium/long) gets its own encoder with a
    learnable embedding that helps the model distinguish between resolutions.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        patch_size: int,
        d_model: int
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, n_features, d_model)
        # Learnable resolution-specific embedding
        self.resolution_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)

        Returns:
            Patch embeddings with resolution encoding of shape (batch, n_patches, d_model)
        """
        patches = self.patch_embed(x)
        return patches + self.resolution_embed


class QuantileHead(nn.Module):
    """
    Prediction head for quantile regression.

    Outputs quantile predictions for each horizon separately,
    allowing horizon-specific learned representations.
    """

    def __init__(self, d_model: int, n_horizons: int, n_quantiles: int):
        super().__init__()
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_quantiles)
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pooled transformer output of shape (batch, d_model)

        Returns:
            Quantile predictions of shape (batch, n_horizons, n_quantiles)
        """
        outputs = [head(x) for head in self.horizon_heads]
        return torch.stack(outputs, dim=1)


class VolumeHead(nn.Module):
    """
    Prediction head for volume quantiles.

    Predicts buy/sell volume quantiles (p10, p50, p90) at each horizon.
    Output shape: (batch, n_horizons, 2, n_quantiles) where dim 2 is [buy, sell].
    """

    def __init__(
        self,
        d_model: int = 384,
        hidden_dim: int = 64,
        n_horizons: int = 7,
        n_quantiles: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

        # Single MLP that outputs all horizons × buy/sell × quantiles
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons * 2 * n_quantiles),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize with smaller weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: Pooled transformer output of shape (batch, d_model)

        Returns:
            Volume predictions of shape (batch, n_horizons, 2, n_quantiles)
        """
        x = self.mlp(pooled)
        return x.view(-1, self.n_horizons, 2, self.n_quantiles)


class PatchTSTModel(nn.Module):
    """
    Multi-resolution Patch Time Series Transformer.

    Architecture:
    1. Three resolution encoders convert input sequences to patch embeddings
    2. Optional item embedding token prepended to sequence
    3. Transformer encoder processes all patches together
    4. Mean pooling + layer norm produces final representation
    5. Separate quantile heads for high and low price predictions
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        # Resolution encoders
        self.recent_encoder = ResolutionEncoder(
            c.recent_len, c.recent_features, c.patch_size, c.d_model
        )
        self.medium_encoder = ResolutionEncoder(
            c.medium_len, c.medium_features, c.patch_size, c.d_model
        )
        self.long_encoder = ResolutionEncoder(
            c.long_len, c.long_features, c.patch_size, c.d_model
        )

        # Calculate total patches for positional encoding
        total_patches = (
            c.recent_len // c.patch_size +
            c.medium_len // c.patch_size +
            c.long_len // c.patch_size
        )
        self.pos_encoder = PositionalEncoding(
            c.d_model, total_patches + 10, c.dropout
        )

        # Item embedding (for multi-item training)
        self.item_embedding = nn.Embedding(c.n_items, c.item_embed_dim)
        self.item_proj = nn.Linear(c.item_embed_dim, c.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=c.n_layers
        )

        # Output heads
        self.pool_norm = nn.LayerNorm(c.d_model)
        self.high_head = QuantileHead(c.d_model, c.n_horizons, c.n_quantiles)
        self.low_head = QuantileHead(c.d_model, c.n_horizons, c.n_quantiles)

        # Optional volume head
        self.volume_head = None
        if c.enable_volume_head:
            self.volume_head = VolumeHead(
                d_model=c.d_model,
                hidden_dim=c.volume_hidden_dim,
                n_horizons=c.n_horizons,
                n_quantiles=len(c.volume_quantiles),
                dropout=c.dropout
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        recent: torch.Tensor,
        medium: torch.Tensor,
        long: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            recent: Recent 5-min features (batch, 288, 6)
            medium: Medium 1-hour features (batch, 168, 10)
            long: Long 4-hour features (batch, 180, 10)
            item_ids: Optional item indices (batch,) for item embeddings

        Returns:
            Dictionary with 'high_quantiles' and 'low_quantiles',
            each of shape (batch, n_horizons, n_quantiles)
        """
        # Encode each resolution
        recent_patches = self.recent_encoder(recent)
        medium_patches = self.medium_encoder(medium)
        long_patches = self.long_encoder(long)

        # Concatenate all patches
        all_patches = torch.cat(
            [recent_patches, medium_patches, long_patches], dim=1
        )

        # Add item token if provided
        if item_ids is not None:
            item_embed = self.item_embedding(item_ids)
            item_token = self.item_proj(item_embed).unsqueeze(1)
            all_patches = torch.cat([item_token, all_patches], dim=1)

        # Apply positional encoding and transformer
        x = self.pos_encoder(all_patches)
        x = self.transformer(x)

        # Mean pooling over sequence
        x = x.mean(dim=1)
        x = self.pool_norm(x)

        # Predict quantiles
        high_q = self.high_head(x)
        low_q = self.low_head(x)

        outputs = {
            'high_quantiles': high_q,
            'low_quantiles': low_q
        }

        # Optional volume prediction
        if self.volume_head is not None:
            outputs['volume_pred'] = self.volume_head(x)

        return outputs


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
) -> torch.Tensor:
    """
    Compute pinball (quantile) loss.

    The pinball loss penalizes under-predictions and over-predictions
    asymmetrically based on the target quantile.

    Args:
        pred: Predicted quantiles of shape (batch, n_horizons, n_quantiles)
        target: Target values of shape (batch, n_horizons) or (batch, n_horizons, 1)
        quantiles: Tuple of quantile levels

    Returns:
        Scalar loss tensor
    """
    if target.dim() == 2:
        target = target.unsqueeze(-1)

    # Expand target to match pred shape
    target = target.expand_as(pred)

    errors = target - pred
    losses = []

    for i, q in enumerate(quantiles):
        error = errors[..., i]
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss)

    return torch.mean(torch.stack(losses))


def combined_quantile_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
) -> torch.Tensor:
    """
    Combined loss for high and low price predictions.

    Args:
        outputs: Model outputs with 'high_quantiles' and 'low_quantiles'
        targets: Target tensor of shape (batch, n_horizons, 2) with (max_high, min_low)
        quantiles: Tuple of quantile levels

    Returns:
        Combined scalar loss tensor
    """
    high_target = targets[..., 0]  # max high price movement
    low_target = targets[..., 1]   # min low price movement

    high_loss = quantile_loss(outputs['high_quantiles'], high_target, quantiles)
    low_loss = quantile_loss(outputs['low_quantiles'], low_target, quantiles)

    return high_loss + low_loss


def combined_loss_with_volume(
    outputs: Dict[str, torch.Tensor],
    price_targets: torch.Tensor,
    volume_targets: Optional[torch.Tensor],
    config: 'ModelConfig',
    volume_weight: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for price and volume prediction.

    Computes quantile loss for both price predictions and volume predictions,
    with volume loss weighted by volume_weight.

    Args:
        outputs: Model outputs with 'high_quantiles', 'low_quantiles', and optionally 'volume_pred'
        price_targets: Price targets of shape (batch, n_horizons, 2) with (max_high_pct, min_low_pct)
        volume_targets: Volume targets of shape (batch, n_horizons, 2) with (buy_vol, sell_vol), or None
        config: ModelConfig with quantile settings
        volume_weight: Weight for volume loss (default 0.2)

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains
        'price_loss', 'volume_loss', and 'total_loss'
    """
    # Price loss (high + low quantile loss)
    price_quantiles = config.quantiles
    high_target = price_targets[..., 0]
    low_target = price_targets[..., 1]

    high_loss = quantile_loss(outputs['high_quantiles'], high_target, price_quantiles)
    low_loss = quantile_loss(outputs['low_quantiles'], low_target, price_quantiles)
    price_loss = high_loss + low_loss

    loss_dict = {'price_loss': price_loss.detach().item()}
    total = price_loss

    # Volume loss (if enabled and available)
    if config.enable_volume_head and 'volume_pred' in outputs and volume_targets is not None:
        vol_pred = outputs['volume_pred']  # (batch, n_horizons, 2, n_vol_quantiles)
        vol_quantiles = config.volume_quantiles

        # Buy side: vol_pred[..., 0, :] vs volume_targets[..., 0]
        # Sell side: vol_pred[..., 1, :] vs volume_targets[..., 1]
        buy_vol_loss = quantile_loss(vol_pred[..., 0, :], volume_targets[..., 0], vol_quantiles)
        sell_vol_loss = quantile_loss(vol_pred[..., 1, :], volume_targets[..., 1], vol_quantiles)
        vol_loss = buy_vol_loss + sell_vol_loss

        total = total + volume_weight * vol_loss
        loss_dict['volume_loss'] = vol_loss.detach().item()
    else:
        loss_dict['volume_loss'] = 0.0

    loss_dict['total_loss'] = total.detach().item()

    return total, loss_dict
