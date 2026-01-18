"""
Tests for High-Value Item Selection (Issue #120)

Tests the configuration loading, item selection logic, and scoring
for high-value items in the training pipeline.
"""

import sys
import os

# Mock the database module before importing anything else
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set dummy environment variables for testing
os.environ.setdefault('DB_PASS', 'test_password')
os.environ.setdefault('DB_HOST', 'localhost')

import pytest
from dataclasses import asdict
from unittest.mock import patch, MagicMock

# Import the modules under test
from training.item_selector import (
    HighValueConfig,
    SelectionConfig,
    SelectedItem,
    SelectionResult,
    ItemSelector,
)


class TestHighValueConfig:
    """Tests for HighValueConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HighValueConfig()
        assert config.enabled is True
        assert config.min_price_gp == 10_000_000  # 10M gp
        assert config.min_24h_volume == 100
        assert config.min_training_rows == 5000
        assert config.max_items_per_run == 20
        assert config.min_history_days == 30
        assert config.force_include_items == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HighValueConfig(
            enabled=False,
            min_price_gp=100_000_000,
            min_24h_volume=50,
            max_items_per_run=10,
        )
        assert config.enabled is False
        assert config.min_price_gp == 100_000_000
        assert config.min_24h_volume == 50
        assert config.max_items_per_run == 10

    def test_priority_weights(self):
        """Test priority weight defaults."""
        config = HighValueConfig()
        assert config.weight_no_model == 100
        assert config.weight_high_price == 50
        assert config.weight_low_manipulation_risk == 30


class TestSelectionConfig:
    """Tests for SelectionConfig with high-value support."""

    def test_default_includes_high_value(self):
        """Test that default config includes high-value config."""
        config = SelectionConfig()
        assert hasattr(config, 'high_value')
        assert isinstance(config.high_value, HighValueConfig)
        assert config.high_value.enabled is True

    def test_high_value_config_serialization(self):
        """Test that config can be serialized to dict."""
        config = SelectionConfig()
        config_dict = asdict(config)
        assert 'high_value' in config_dict
        assert config_dict['high_value']['enabled'] is True
        assert config_dict['high_value']['min_price_gp'] == 10_000_000


class TestSelectedItem:
    """Tests for SelectedItem with high-value flag."""

    def test_default_not_high_value(self):
        """Test that default items are not high-value."""
        item = SelectedItem(
            item_id=123,
            item_name="Test Item",
            reason="test",
            priority_score=100,
        )
        assert item.is_high_value is False
        assert item.item_price is None

    def test_high_value_item(self):
        """Test high-value item attributes."""
        item = SelectedItem(
            item_id=20997,
            item_name="Twisted bow",
            reason="high_value,no_model",
            priority_score=150,
            is_high_value=True,
            item_price=1_400_000_000,
        )
        assert item.is_high_value is True
        assert item.item_price == 1_400_000_000

    def test_serialization(self):
        """Test that SelectedItem serializes correctly."""
        item = SelectedItem(
            item_id=20997,
            item_name="Twisted bow",
            reason="high_value,no_model",
            priority_score=150,
            is_high_value=True,
            item_price=1_400_000_000,
        )
        item_dict = asdict(item)
        assert item_dict['is_high_value'] is True
        assert item_dict['item_price'] == 1_400_000_000


class TestSelectionResult:
    """Tests for SelectionResult with high-value tracking."""

    def test_high_value_counts(self):
        """Test high-value item counting in results."""
        result = SelectionResult(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            config={},
            items=[],
            total_eligible=100,
            total_selected=10,
            selection_reasons={'no_model': 5, 'high_value': 3},
            high_value_selected=3,
            high_value_eligible=5,
        )
        assert result.high_value_selected == 3
        assert result.high_value_eligible == 5


class TestItemSelectorHighValue:
    """Tests for ItemSelector high-value functionality."""

    @pytest.fixture
    def selector(self):
        """Create a selector with mocked database."""
        selector = ItemSelector(config=SelectionConfig())
        return selector

    def test_score_high_value_item_no_model(self, selector):
        """Test scoring for high-value item without model."""
        item = {
            'item_id': 20997,
            'item_name': 'Twisted bow',
            'item_price': 1_400_000_000,
            'model_id': None,
        }
        score, reason = selector._score_high_value_item(item)

        # Should get no_model weight (100) + price weight
        assert score >= 100
        assert 'no_model' in reason
        assert 'price' in reason

    def test_score_high_value_item_with_model(self, selector):
        """Test scoring for high-value item with existing model."""
        item = {
            'item_id': 20997,
            'item_name': 'Twisted bow',
            'item_price': 1_400_000_000,
            'model_id': 123,
        }
        score, reason = selector._score_high_value_item(item)

        # Should only get price weight, not no_model
        assert score < 100
        assert 'no_model' not in reason
        assert 'price' in reason

    def test_score_high_value_item_price_scaling(self, selector):
        """Test that higher prices get higher scores."""
        item_low = {
            'item_id': 1,
            'item_name': 'Cheap Item',
            'item_price': 10_000_000,  # 10M
            'model_id': None,
        }
        item_high = {
            'item_id': 2,
            'item_name': 'Expensive Item',
            'item_price': 1_000_000_000,  # 1B
            'model_id': None,
        }

        score_low, _ = selector._score_high_value_item(item_low)
        score_high, _ = selector._score_high_value_item(item_high)

        # Higher price should have higher score
        assert score_high > score_low

    def test_select_items_include_high_value_flag(self, selector):
        """Test that include_high_value parameter works."""
        # Mock the database methods
        selector._get_candidate_items = MagicMock(return_value=[
            {'item_id': 1, 'item_name': 'Regular Item', 'model_id': None}
        ])
        selector._get_high_value_candidate_items = MagicMock(return_value=[
            {'item_id': 2, 'item_name': 'High Value Item', 'item_price': 100_000_000, 'model_id': None}
        ])

        # Test with high-value included
        items = selector.select_items_for_training(max_items=10, include_high_value=True)
        selector._get_high_value_candidate_items.assert_called_once()

        # Test with high-value excluded
        selector._get_high_value_candidate_items.reset_mock()
        items = selector.select_items_for_training(max_items=10, include_high_value=False)
        selector._get_high_value_candidate_items.assert_not_called()

    def test_select_items_high_value_only(self, selector):
        """Test high_value_only mode."""
        selector._get_candidate_items = MagicMock(return_value=[
            {'item_id': 1, 'item_name': 'Regular Item', 'model_id': None}
        ])
        selector._get_high_value_candidate_items = MagicMock(return_value=[
            {'item_id': 2, 'item_name': 'High Value Item', 'item_price': 100_000_000, 'model_id': None}
        ])

        items = selector.select_items_for_training(max_items=10, high_value_only=True)

        # Should not call regular candidate method
        selector._get_candidate_items.assert_not_called()
        # Should call high-value method
        selector._get_high_value_candidate_items.assert_called_once()

    def test_force_include_high_value_items(self, selector):
        """Test force_include_items in high-value config."""
        selector.config.high_value.force_include_items = [20997]
        selector._get_candidate_items = MagicMock(return_value=[])
        selector._get_high_value_candidate_items = MagicMock(return_value=[])
        selector._get_item_info = MagicMock(return_value={
            'item_id': 20997, 'name': 'Twisted bow'
        })

        items = selector.select_items_for_training(max_items=10)

        # Force-included item should be in results
        assert len(items) == 1
        assert items[0].item_id == 20997
        assert items[0].is_high_value is True
        assert 'forced_high_value' in items[0].reason


class TestConfigLoading:
    """Tests for loading high-value config from YAML."""

    def test_load_config_with_high_value(self, tmp_path):
        """Test loading config file with high-value section."""
        config_content = """
item_selection:
  min_24h_volume: 10000
  min_training_rows: 5000
  max_items_per_run: 50

  high_value:
    enabled: true
    min_price_gp: 50000000
    min_24h_volume: 200
    max_items_per_run: 15
    priority_weights:
      no_model: 100
      high_price: 60
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        selector = ItemSelector(config_path=str(config_path))

        assert selector.config.high_value.enabled is True
        assert selector.config.high_value.min_price_gp == 50_000_000
        assert selector.config.high_value.min_24h_volume == 200
        assert selector.config.high_value.max_items_per_run == 15
        assert selector.config.high_value.weight_high_price == 60

    def test_load_config_without_high_value(self, tmp_path):
        """Test loading config file without high-value section uses defaults."""
        config_content = """
item_selection:
  min_24h_volume: 10000
  min_training_rows: 5000
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        selector = ItemSelector(config_path=str(config_path))

        # Should use default high-value config
        assert selector.config.high_value.enabled is True
        assert selector.config.high_value.min_price_gp == 10_000_000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
