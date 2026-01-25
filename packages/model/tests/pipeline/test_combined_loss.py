# packages/model/tests/pipeline/test_combined_loss.py
import torch
import pytest


def test_combined_loss_with_volume_basic():
    """Combined loss computes price + weighted volume loss."""
    from src.pipeline.model import combined_loss_with_volume
    from src.pipeline.config import ModelConfig

    config = ModelConfig(enable_volume_head=True)
    batch_size = 8
    n_horizons = 7
    n_price_quantiles = 5
    n_volume_quantiles = 3

    outputs = {
        'high_quantiles': torch.randn(batch_size, n_horizons, n_price_quantiles),
        'low_quantiles': torch.randn(batch_size, n_horizons, n_price_quantiles),
        'volume_pred': torch.randn(batch_size, n_horizons, 2, n_volume_quantiles),
    }
    price_targets = torch.randn(batch_size, n_horizons, 2)
    volume_targets = torch.randn(batch_size, n_horizons, 2)

    loss, loss_dict = combined_loss_with_volume(
        outputs, price_targets, volume_targets, config, volume_weight=0.2
    )

    assert loss.dim() == 0  # scalar
    assert 'price_loss' in loss_dict
    assert 'volume_loss' in loss_dict
    assert 'total_loss' in loss_dict
    assert loss_dict['total_loss'] > 0


def test_combined_loss_without_volume():
    """Combined loss works when volume head is disabled."""
    from src.pipeline.model import combined_loss_with_volume
    from src.pipeline.config import ModelConfig

    config = ModelConfig(enable_volume_head=False)
    batch_size = 8
    n_horizons = 7
    n_price_quantiles = 5

    outputs = {
        'high_quantiles': torch.randn(batch_size, n_horizons, n_price_quantiles),
        'low_quantiles': torch.randn(batch_size, n_horizons, n_price_quantiles),
    }
    price_targets = torch.randn(batch_size, n_horizons, 2)
    volume_targets = None  # No volume targets

    loss, loss_dict = combined_loss_with_volume(
        outputs, price_targets, volume_targets, config, volume_weight=0.2
    )

    assert loss.dim() == 0  # scalar
    assert 'price_loss' in loss_dict
    assert loss_dict['volume_loss'] == 0.0  # No volume loss
    assert loss_dict['price_loss'] == loss_dict['total_loss']


def test_combined_loss_volume_weight():
    """Volume weight correctly scales volume loss contribution."""
    from src.pipeline.model import combined_loss_with_volume
    from src.pipeline.config import ModelConfig

    config = ModelConfig(enable_volume_head=True)

    torch.manual_seed(42)
    batch_size = 8
    n_horizons = 7

    outputs = {
        'high_quantiles': torch.randn(batch_size, n_horizons, 5),
        'low_quantiles': torch.randn(batch_size, n_horizons, 5),
        'volume_pred': torch.randn(batch_size, n_horizons, 2, 3),
    }
    price_targets = torch.randn(batch_size, n_horizons, 2)
    volume_targets = torch.randn(batch_size, n_horizons, 2)

    # Test with different weights
    _, dict_low = combined_loss_with_volume(
        outputs, price_targets, volume_targets, config, volume_weight=0.1
    )
    _, dict_high = combined_loss_with_volume(
        outputs, price_targets, volume_targets, config, volume_weight=0.5
    )

    # Same price loss
    assert abs(dict_low['price_loss'] - dict_high['price_loss']) < 1e-5

    # Same raw volume loss
    assert abs(dict_low['volume_loss'] - dict_high['volume_loss']) < 1e-5

    # Total differs by weight contribution
    expected_diff = (0.5 - 0.1) * dict_low['volume_loss']
    actual_diff = dict_high['total_loss'] - dict_low['total_loss']
    assert abs(expected_diff - actual_diff) < 1e-5
