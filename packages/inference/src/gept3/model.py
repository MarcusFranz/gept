from __future__ import annotations

import torch
import torch.nn as nn

from .config import GePT3ModelConfig


class _GridHead(nn.Module):
    """Per-item grid head over (horizon, offset)."""

    def __init__(self, d_model: int, n_horizons: int, n_offsets: int, dropout: float):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_offsets = n_offsets
        self.horizon_embedding = nn.Embedding(n_horizons, d_model)
        self.offset_embedding = nn.Embedding(n_offsets, d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.register_buffer("_h_idx", torch.arange(n_horizons, dtype=torch.long), persistent=False)
        self.register_buffer("_o_idx", torch.arange(n_offsets, dtype=torch.long), persistent=False)

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        # token_repr: (batch, items, d_model)
        h = self.horizon_embedding(self._h_idx)  # (H, D)
        o = self.offset_embedding(self._o_idx)  # (O, D)
        z = (
            token_repr.unsqueeze(2).unsqueeze(3)
            + h.unsqueeze(0).unsqueeze(0).unsqueeze(3)
            + o.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        )  # (B, I, H, O, D)
        return self.proj(z).squeeze(-1)  # (B, I, H, O)


class _PriceHead(nn.Module):
    """Per-item horizon head for high/low quantile deltas."""

    def __init__(self, d_model: int, n_horizons: int, n_price_quantiles: int, dropout: float):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_price_quantiles = n_price_quantiles
        self.horizon_embedding = nn.Embedding(n_horizons, d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * n_price_quantiles),
        )
        self.register_buffer("_h_idx", torch.arange(n_horizons, dtype=torch.long), persistent=False)

    def forward(self, token_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.horizon_embedding(self._h_idx)  # (H, D)
        z = token_repr.unsqueeze(2) + h.unsqueeze(0).unsqueeze(0)  # (B, I, H, D)
        out = self.proj(z)  # (B, I, H, 2Q)
        hi, lo = torch.split(out, self.n_price_quantiles, dim=-1)
        return hi, lo


class GePT3UnifiedModel(nn.Module):
    """GePT3 unified architecture with cross-item attention and multi-head outputs."""

    def __init__(self, config: GePT3ModelConfig):
        super().__init__()
        self.n_items = int(config.n_items)
        self.n_features = int(config.n_features)
        self.n_horizons = int(config.n_horizons)
        self.n_offsets = int(config.n_offsets)

        self.feature_proj = nn.Sequential(
            nn.Linear(config.n_features, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=int(config.ff_mult * config.d_model),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_layers)
        self.output_norm = nn.LayerNorm(config.d_model)

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

    def encode(self, x: torch.Tensor, item_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode item tokens with cross-item self-attention."""
        # x: (B, I, F), item_id: (B, I), mask: (B, I) where True means valid.
        item_id = item_id.long().clamp_(0, self.n_items - 1)
        z = self.feature_proj(x) + self.item_embedding(item_id)
        z = self.input_dropout(z)
        key_padding_mask = ~mask.to(dtype=torch.bool)
        z = self.encoder(z, src_key_padding_mask=key_padding_mask)
        z = self.output_norm(z)
        return z

    def forward(self, x: torch.Tensor, item_id: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        token_repr = self.encode(x=x, item_id=item_id, mask=mask)
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
