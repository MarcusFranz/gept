# packages/model/tests/pipeline/test_volume_head.py
import torch
import pytest


def test_volume_head_output_shape():
    """VolumeHead produces correct output shape."""
    from src.pipeline.model import VolumeHead

    batch_size = 32
    d_model = 384
    n_horizons = 7
    n_quantiles = 3

    head = VolumeHead(d_model=d_model, n_horizons=n_horizons, n_quantiles=n_quantiles)
    pooled = torch.randn(batch_size, d_model)

    output = head(pooled)

    # Shape should be (batch, n_horizons, 2, n_quantiles) for buy/sell
    assert output.shape == (batch_size, n_horizons, 2, n_quantiles)


def test_volume_head_default_quantiles():
    """VolumeHead uses sensible defaults."""
    from src.pipeline.model import VolumeHead

    head = VolumeHead()
    pooled = torch.randn(8, 384)

    output = head(pooled)

    # Default: 7 horizons, 3 quantiles (p10, p50, p90)
    assert output.shape == (8, 7, 2, 3)


def test_model_with_volume_head():
    """PatchTSTModel includes volume predictions when enabled."""
    from src.pipeline.model import PatchTSTModel
    from src.pipeline.config import ModelConfig

    config = ModelConfig(enable_volume_head=True, n_items=100)
    model = PatchTSTModel(config)

    batch_size = 4
    recent = torch.randn(batch_size, config.recent_len, config.recent_features)
    medium = torch.randn(batch_size, config.medium_len, config.medium_features)
    long = torch.randn(batch_size, config.long_len, config.long_features)
    item_ids = torch.randint(0, 100, (batch_size,))

    outputs = model(recent, medium, long, item_ids)

    assert 'high_quantiles' in outputs
    assert 'low_quantiles' in outputs
    assert 'volume_pred' in outputs
    assert outputs['volume_pred'].shape == (batch_size, config.n_horizons, 2, 3)


def test_model_without_volume_head():
    """PatchTSTModel excludes volume predictions when disabled."""
    from src.pipeline.model import PatchTSTModel
    from src.pipeline.config import ModelConfig

    config = ModelConfig(enable_volume_head=False, n_items=100)
    model = PatchTSTModel(config)

    batch_size = 4
    recent = torch.randn(batch_size, config.recent_len, config.recent_features)
    medium = torch.randn(batch_size, config.medium_len, config.medium_features)
    long = torch.randn(batch_size, config.long_len, config.long_features)

    outputs = model(recent, medium, long)

    assert 'high_quantiles' in outputs
    assert 'low_quantiles' in outputs
    assert 'volume_pred' not in outputs
