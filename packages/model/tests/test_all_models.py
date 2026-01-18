"""
Basic Function Tests for All Trained Models

Verifies:
1. All model files exist and load correctly
2. Models return probabilities in [0, 1]
3. No NaN/Inf in outputs
4. Inference time is reasonable (<20ms per model)
"""

import sys
import os
import time
import json
import joblib
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


MODELS_DIR = Path(__file__).parent.parent / 'models_downloaded'


def get_all_model_files():
    """Find all model files in the models directory."""
    if not MODELS_DIR.exists():
        return []

    model_files = []
    for item_dir in MODELS_DIR.iterdir():
        if not item_dir.is_dir():
            continue
        for f in item_dir.iterdir():
            if f.name.endswith('_model.pkl'):
                model_files.append(f)
    return model_files


def get_all_item_registries():
    """Find all item registry files."""
    if not MODELS_DIR.exists():
        return []

    registries = []
    for item_dir in MODELS_DIR.iterdir():
        if not item_dir.is_dir():
            continue
        registry = item_dir / 'registry.json'
        if registry.exists():
            registries.append(registry)
    return registries


class TestModelLoading:
    """Test that all models load correctly."""

    def test_models_directory_exists(self):
        """Check that models directory exists (skips in CI without models)."""
        if not MODELS_DIR.exists():
            pytest.skip(f"Models directory not found: {MODELS_DIR} (expected in CI)")

    def test_at_least_one_item_trained(self):
        """Check that at least one item has trained models."""
        registries = get_all_item_registries()
        if len(registries) == 0:
            pytest.skip("No trained items found (expected in CI without models)")

    @pytest.mark.parametrize("model_file", get_all_model_files()[:50])  # Test first 50
    def test_model_loads(self, model_file):
        """Test that each model file loads without error."""
        model = joblib.load(model_file)
        assert model is not None

    @pytest.mark.parametrize("model_file", get_all_model_files()[:50])
    def test_scaler_exists(self, model_file):
        """Test that corresponding scaler file exists."""
        scaler_file = model_file.parent / model_file.name.replace('_model.pkl', '_scaler.pkl')
        assert scaler_file.exists(), f"Scaler not found: {scaler_file}"


class TestModelPredictions:
    """Test that models produce valid predictions."""

    @pytest.fixture
    def sample_features(self):
        """Generate random feature vector for testing."""
        # Typical feature count is ~80
        return np.random.randn(1, 80)

    def test_prediction_returns_probability(self, sample_features):
        """Test that predictions are probabilities in [0, 1]."""
        model_files = get_all_model_files()
        if not model_files:
            pytest.skip("No models found")

        model_file = model_files[0]
        scaler_file = model_file.parent / model_file.name.replace('_model.pkl', '_scaler.pkl')
        meta_file = model_file.parent / model_file.name.replace('_model.pkl', '_meta.json')

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        with open(meta_file) as f:
            meta = json.load(f)

        # Create properly sized feature vector
        n_features = len(meta.get('feature_columns', []))
        if n_features == 0:
            n_features = 80

        X = np.random.randn(1, n_features)
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)[0, 1]

        assert 0 <= proba <= 1, f"Probability {proba} out of range [0, 1]"
        assert not np.isnan(proba), "Probability is NaN"
        assert not np.isinf(proba), "Probability is Inf"

    def test_inference_time(self):
        """Test that inference is fast (<20ms per model)."""
        model_files = get_all_model_files()
        if len(model_files) < 5:
            pytest.skip("Not enough models for timing test")

        times = []
        for model_file in model_files[:10]:
            scaler_file = model_file.parent / model_file.name.replace('_model.pkl', '_scaler.pkl')
            meta_file = model_file.parent / model_file.name.replace('_model.pkl', '_meta.json')

            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)

            with open(meta_file) as f:
                meta = json.load(f)

            n_features = len(meta.get('feature_columns', []))
            if n_features == 0:
                n_features = 80

            X = np.random.randn(1, n_features)

            start = time.time()
            X_scaled = scaler.transform(X)
            _ = model.predict_proba(X_scaled)
            elapsed = time.time() - start

            times.append(elapsed * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        assert avg_time < 20, f"Average inference time {avg_time:.1f}ms exceeds 20ms"


class TestModelMetadata:
    """Test that model metadata is complete."""

    @pytest.mark.parametrize("registry_file", get_all_item_registries()[:20])
    def test_registry_has_required_fields(self, registry_file):
        """Test that registry has required fields."""
        with open(registry_file) as f:
            registry = json.load(f)

        assert 'item_id' in registry
        assert 'models' in registry
        assert len(registry['models']) > 0

    @pytest.mark.parametrize("registry_file", get_all_item_registries()[:20])
    def test_model_metadata_complete(self, registry_file):
        """Test that each model has complete metadata."""
        with open(registry_file) as f:
            registry = json.load(f)

        for target_name in list(registry['models'].keys())[:5]:
            meta_file = registry_file.parent / f'{target_name}_meta.json'
            assert meta_file.exists(), f"Meta file missing: {meta_file}"

            with open(meta_file) as f:
                meta = json.load(f)

            assert 'feature_columns' in meta
            assert 'metrics' in meta
            assert 'auc' in meta['metrics']


class TestModelCoverage:
    """Test overall model coverage."""

    def test_minimum_items_trained(self):
        """Check that minimum number of items are trained."""
        registries = get_all_item_registries()
        if len(registries) == 0:
            pytest.skip("No models found (expected in CI without models)")
        # Expect at least 100 items (out of 108) when models are present
        assert len(registries) >= 100, f"Only {len(registries)} items trained"

    def test_minimum_models_per_item(self):
        """Check that each item has reasonable model count."""
        registries = get_all_item_registries()
        if not registries:
            pytest.skip("No registries found")

        model_counts = []
        for registry_file in registries:
            with open(registry_file) as f:
                registry = json.load(f)
            model_counts.append(len(registry.get('models', {})))

        avg_models = sum(model_counts) / len(model_counts)
        # Expect at least 10 models per item on average
        assert avg_models >= 10, f"Average models per item ({avg_models:.1f}) is too low"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
