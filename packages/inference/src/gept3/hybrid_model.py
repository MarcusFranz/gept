from __future__ import annotations

import torch
import torch.nn as nn

from .config import GePT3ModelConfig
from .model import _GridHead, _PriceHead


class GePT3HybridModel(nn.Module):
    """Hybrid GePT3: per-item temporal encoder + cross-item attention."""

    def __init__(self, config: GePT3ModelConfig):
        super().__init__()
        if config.temporal_len <= 0:
            raise ValueError("Hybrid model requires config.temporal_len > 0")

        self.n_items = int(config.n_items)
        self.n_features = int(config.n_features)
        self.n_horizons = int(config.n_horizons)
        self.n_offsets = int(config.n_offsets)
        self.temporal_len = int(config.temporal_len)

        self.temporal_feature_proj = nn.Sequential(
            nn.Linear(config.n_features, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.temporal_pos_embedding = nn.Embedding(self.temporal_len, config.d_model)
        self.register_buffer("_t_idx", torch.arange(self.temporal_len, dtype=torch.long), persistent=False)
        self.temporal_dropout = nn.Dropout(config.dropout)

        temporal_enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.temporal_n_heads,
            dim_feedforward=int(config.ff_mult * config.d_model),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_enc_layer, num_layers=config.temporal_n_layers)
        self.temporal_out_norm = nn.LayerNorm(config.d_model)

        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.item_input_dropout = nn.Dropout(config.dropout)

        cross_enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=int(config.ff_mult * config.d_model),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_item_encoder = nn.TransformerEncoder(cross_enc_layer, num_layers=config.n_layers)
        self.cross_out_norm = nn.LayerNorm(config.d_model)

        self.sell_head = (
            _GridHead(config.d_model, config.n_horizons, config.n_offsets, config.dropout)
            if config.enable_sell_head
            else None
        )
        self.buy_head = (
            _GridHead(config.d_model, config.n_horizons, config.n_offsets, config.dropout)
            if config.enable_buy_head
            else None
        )
        self.price_head = (
            _PriceHead(config.d_model, config.n_horizons, config.n_price_quantiles, config.dropout)
            if config.enable_price_head
            else None
        )
        self.regime_head = (
            nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 1),
            )
            if config.enable_regime_head
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _encode_temporal(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode per-item temporal history into one token per item."""
        # x_seq: (B, I, T, F)
        b, i, t, _ = x_seq.shape
        if t != self.temporal_len:
            raise ValueError(f"Expected temporal_len={self.temporal_len}, got {t}")

        x_flat = x_seq.reshape(b * i, t, -1)
        z = self.temporal_feature_proj(x_flat)
        z = z + self.temporal_pos_embedding(self._t_idx).unsqueeze(0)
        z = self.temporal_dropout(z)
        z = self.temporal_encoder(z)
        z = self.temporal_out_norm(z)

        # Last timestep summary token.
        item_repr = z[:, -1, :].reshape(b, i, -1)
        return item_repr

    def encode(self, x_seq: torch.Tensor, item_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode temporal + cross-item context."""
        item_id = item_id.long().clamp_(0, self.n_items - 1)
        z = self._encode_temporal(x_seq) + self.item_embedding(item_id)
        z = self.item_input_dropout(z)

        key_padding_mask = ~mask.to(dtype=torch.bool)
        z = self.cross_item_encoder(z, src_key_padding_mask=key_padding_mask)
        z = self.cross_out_norm(z)
        return z

    def forward(self, x_seq: torch.Tensor, item_id: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        token_repr = self.encode(x_seq=x_seq, item_id=item_id, mask=mask)
        out: dict[str, torch.Tensor] = {"token_repr": token_repr}

        if self.sell_head is not None:
            out["sell_logits"] = self.sell_head(token_repr)
        if self.buy_head is not None:
            out["buy_logits"] = self.buy_head(token_repr)
        if self.price_head is not None:
            hi, lo = self.price_head(token_repr)
            out["price_high_deltas"] = hi
            out["price_low_deltas"] = lo
        if self.regime_head is not None:
            out["regime_logit"] = self.regime_head(token_repr).squeeze(-1)

        return out
