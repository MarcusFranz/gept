"""Tests for the model runner."""

from src.model_runner import ModelRunner


class TestModelRunner:
    """Test cases for ModelRunner."""

    def test_init_with_valid_registry(self, sample_registry):
        """Test initialization with valid registry."""
        runner = ModelRunner(str(sample_registry))

        assert runner.registry is not None
        assert "items" in runner.registry
        assert "554" in runner.registry["items"]

    def test_init_with_missing_registry(self, tmp_path):
        """Test initialization with missing registry file."""
        runner = ModelRunner(str(tmp_path / "missing.json"))

        assert runner.registry == {"metadata": {}, "items": {}}

    def test_get_supported_items(self, sample_registry):
        """Test getting list of supported items."""
        runner = ModelRunner(str(sample_registry))
        items = runner.get_supported_items()

        assert 554 in items
        assert 565 in items
        assert len(items) == 2

    def test_get_item_metadata(self, sample_registry):
        """Test getting item metadata."""
        runner = ModelRunner(str(sample_registry))
        metadata = runner.get_item_metadata(554)

        assert metadata is not None
        assert metadata["item_name"] == "Fire rune"
        assert metadata["tier"] == 1
        assert metadata["avg_auc"] == 0.90

    def test_get_item_metadata_missing_item(self, sample_registry):
        """Test getting metadata for missing item."""
        runner = ModelRunner(str(sample_registry))
        metadata = runner.get_item_metadata(999999)

        assert metadata is None

    def test_get_model_metadata(self, sample_registry):
        """Test getting specific model metadata."""
        runner = ModelRunner(str(sample_registry))
        metadata = runner.get_model_metadata(554, 1, 0.02)

        assert metadata is not None
        assert metadata["auc"] == 0.96
        assert metadata["calibrated"] is True

    def test_get_model_metadata_missing(self, sample_registry):
        """Test getting metadata for missing model."""
        runner = ModelRunner(str(sample_registry))

        # Missing item
        assert runner.get_model_metadata(999999, 1, 0.02) is None

        # Missing config
        assert runner.get_model_metadata(554, 24, 0.05) is None

    def test_get_registry_metadata(self, sample_registry):
        """Test getting registry metadata."""
        runner = ModelRunner(str(sample_registry))
        metadata = runner.get_registry_metadata()

        assert metadata["model_type"] == "catboost"
        assert metadata["total_items"] == 2

    def test_model_key_generation(self, sample_registry):
        """Test model cache key generation."""
        runner = ModelRunner(str(sample_registry))

        key = runner._get_model_key(554, 1, 0.02)
        assert key == "554_1h_2_0pct"

        key = runner._get_model_key(554, 24, 0.015)
        assert key == "554_24h_1_5pct"

    def test_config_key_generation(self, sample_registry):
        """Test registry config key generation."""
        runner = ModelRunner(str(sample_registry))

        key = runner._get_model_config_key(1, 0.02)
        assert key == "1h_2.0pct"

        key = runner._get_model_config_key(24, 0.015)
        assert key == "24h_1.5pct"

    def test_health_check_without_models(self, sample_registry):
        """Test health check with registry but no loaded models."""
        runner = ModelRunner(str(sample_registry))
        health = runner.health_check()

        assert health["status"] == "ok"
        assert health["supported_items"] == 2
        assert health["cached_models"] == 0

    def test_clear_cache(self, sample_registry):
        """Test clearing model cache."""
        runner = ModelRunner(str(sample_registry))

        # Add some dummy entries to cache
        runner.models["test"] = None
        runner.scalers["test"] = None

        runner.clear_cache()

        assert len(runner.models) == 0
        assert len(runner.scalers) == 0

    def test_model_key_generation_new_offsets(self, sample_registry):
        """Test model cache key generation for new granular offset percentages."""
        runner = ModelRunner(str(sample_registry))

        # 1.25% -> formatted as 1.2 (rounds down) -> "1_2pct"
        key = runner._get_model_key(554, 1, 0.0125)
        assert key == "554_1h_1_2pct"

        # 1.75% -> formatted as 1.8 (banker's rounding up) -> "1_8pct"
        key = runner._get_model_key(554, 2, 0.0175)
        assert key == "554_2h_1_8pct"

        # 2.25% -> formatted as 2.2 (rounds down) -> "2_2pct"
        key = runner._get_model_key(554, 4, 0.0225)
        assert key == "554_4h_2_2pct"

        # 2.5% -> formatted as 2.5 -> "2_5pct"
        key = runner._get_model_key(554, 8, 0.0250)
        assert key == "554_8h_2_5pct"

    def test_offset_percentages_constant(self, sample_registry):
        """Test that OFFSET_PERCENTAGES contains all granular offsets."""
        runner = ModelRunner(str(sample_registry))

        expected_offsets = [0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250]
        assert runner.OFFSET_PERCENTAGES == expected_offsets
