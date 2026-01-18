"""
Tests for CatBoost Model Format Conversion

Tests the conversion of CatBoost models from legacy format to native (.cbm) format,
and the predictor's ability to load and use .cbm files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile  # noqa: E402
import pytest  # noqa: E402
import numpy as np  # noqa: E402

from convert_to_cbm import is_catboost_model, convert_pkl_to_cbm  # noqa: E402

try:
    from catboost import CatBoostClassifier  # noqa: E402
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from sklearn.preprocessing import StandardScaler  # noqa: E402
    from sklearn.ensemble import RandomForestClassifier  # noqa: E402
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestModelDetection:
    """Tests for model type detection."""

    def test_is_catboost_model_with_catboost(self):
        """Test CatBoost model detection with actual CatBoost model."""
        if not HAS_CATBOOST:
            pytest.skip("CatBoost not installed")

        model = CatBoostClassifier(iterations=10, verbose=0)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y, verbose=False)

        assert is_catboost_model(model) is True

    def test_is_catboost_model_with_sklearn(self):
        """Test CatBoost model detection returns False for sklearn models."""
        if not HAS_SKLEARN:
            pytest.skip("sklearn not installed")

        model = RandomForestClassifier(n_estimators=10)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)

        assert is_catboost_model(model) is False


@pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not installed")
class TestCbmConversion:
    """Tests for CatBoost .cbm conversion."""

    @pytest.fixture
    def trained_catboost_model(self):
        """Create a trained CatBoost model for testing."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        model = CatBoostClassifier(
            iterations=50,
            depth=3,
            verbose=0,
            random_state=42
        )
        model.fit(X, y, verbose=False)
        return model

    def test_cbm_export_and_load(self, trained_catboost_model):
        """Test that CatBoost model exports to .cbm and loads correctly."""
        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as f:
            output_path = f.name

        try:
            trained_catboost_model.save_model(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            loaded_model = CatBoostClassifier()
            loaded_model.load_model(output_path)

            X_test = np.random.randn(5, 10).astype(np.float32)
            proba_original = trained_catboost_model.predict_proba(X_test)
            proba_loaded = loaded_model.predict_proba(X_test)

            np.testing.assert_allclose(proba_original, proba_loaded)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cbm_prediction_accuracy(self, trained_catboost_model):
        """Test that .cbm model produces valid probability predictions."""
        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as f:
            cbm_path = f.name

        try:
            trained_catboost_model.save_model(cbm_path)

            loaded_model = CatBoostClassifier()
            loaded_model.load_model(cbm_path)

            X_test = np.random.randn(100, 10).astype(np.float32)
            proba = loaded_model.predict_proba(X_test)

            assert proba.shape == (100, 2)
            assert np.all(proba >= 0)
            assert np.all(proba <= 1)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        finally:
            if os.path.exists(cbm_path):
                os.unlink(cbm_path)

    def test_pkl_to_cbm_conversion(self, trained_catboost_model):
        """Test the convert_pkl_to_cbm function."""
        import joblib

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pkl_path = f.name
        cbm_path = pkl_path.replace('.pkl', '.cbm')

        try:
            joblib.dump(trained_catboost_model, pkl_path)

            success = convert_pkl_to_cbm(pkl_path, cbm_path)
            assert success is True
            assert os.path.exists(cbm_path)

            loaded_model = CatBoostClassifier()
            loaded_model.load_model(cbm_path)

            X_test = np.random.randn(5, 10).astype(np.float32)
            proba_original = trained_catboost_model.predict_proba(X_test)
            proba_converted = loaded_model.predict_proba(X_test)

            np.testing.assert_allclose(proba_original, proba_converted)

        finally:
            if os.path.exists(pkl_path):
                os.unlink(pkl_path)
            if os.path.exists(cbm_path):
                os.unlink(cbm_path)


@pytest.mark.skipif(not HAS_CATBOOST or not HAS_SKLEARN,
                    reason="Required dependencies not installed")
class TestEndToEndPrediction:
    """End-to-end tests for the prediction pipeline with .cbm models."""

    def test_cbm_model_with_scaler(self):
        """Test inference with .cbm model and sklearn scaler."""
        import joblib
        import tempfile

        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        model = CatBoostClassifier(iterations=20, depth=3, verbose=0, random_state=42)
        model.fit(X, y, verbose=False)

        scaler = StandardScaler()
        scaler.fit(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            cbm_path = os.path.join(temp_dir, "model.cbm")
            model.save_model(cbm_path)

            scaler_path = os.path.join(temp_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)

            loaded_model = CatBoostClassifier()
            loaded_model.load_model(cbm_path)
            loaded_scaler = joblib.load(scaler_path)

            X_test = np.random.randn(10, 10).astype(np.float32)

            X_scaled = loaded_scaler.transform(X_test)
            proba = loaded_model.predict_proba(X_scaled)

            original_scaled = scaler.transform(X_test)
            original_proba = model.predict_proba(original_scaled)

            np.testing.assert_allclose(proba, original_proba, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
