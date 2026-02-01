"""Test fixtures for the prediction engine."""

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clear_api_key():
    """Ensure API key auth is bypassed in tests.

    Without this, tests fail in environments (e.g. codespaces) where
    INTERNAL_API_KEY is set for production use. We patch both the
    config singleton and os.environ to handle module reloads.
    """
    import src.config as config_module

    original = config_module.config.internal_api_key
    config_module.config.internal_api_key = ""
    with patch.dict(os.environ, {"INTERNAL_API_KEY": ""}, clear=False):
        yield
    config_module.config.internal_api_key = original


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    now = datetime.now(timezone.utc)
    timestamps = pd.date_range(
        end=now,
        periods=1500,  # ~25 hours of 1-min data
        freq="1min",
        tz=timezone.utc,
    )

    np.random.seed(42)
    base_price = 250
    prices = base_price + np.cumsum(np.random.randn(len(timestamps)) * 0.5)
    spreads = np.random.uniform(2, 8, len(timestamps))

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "high": (prices + spreads / 2).astype(int),
            "low": (prices - spreads / 2).astype(int),
            "high_volume": np.random.randint(100, 1000, len(timestamps)),
            "low_volume": np.random.randint(100, 1000, len(timestamps)),
        }
    )

    df = df.set_index("timestamp")
    return df


@pytest.fixture
def sample_features():
    """Generate sample feature vector."""
    return np.array(
        [
            250.0,  # mid
            5.0,  # spread
            0.02,  # spread_pct
            1.001,  # ma_ratio_60
            1.002,  # ma_ratio_240
            1.001,  # ma_ratio_480
            0.999,  # ma_ratio_1440
            0.001,  # return_5
            0.002,  # return_15
            0.003,  # return_30
            0.005,  # return_60
            0.01,  # return_240
            0.005,  # volatility_60
            0.008,  # volatility_240
            0.012,  # volatility_1440
            500.0,  # volume
            450.0,  # volume_ma_1440
            1.11,  # volume_ratio
            0.019,  # spread_ma_240
            1.05,  # spread_ratio
            0.03,  # range_60
            0.05,  # range_240
            0.08,  # range_1440
            14,  # hour
            2,  # day_of_week
            0,  # is_weekend
        ]
    )


@pytest.fixture
def sample_registry(tmp_path):
    """Create a sample model registry."""
    registry = {
        "metadata": {
            "created_at": "2026-01-08T12:00:00Z",
            "model_type": "catboost",
            "total_items": 2,
            "total_models": 14,
        },
        "items": {
            "554": {
                "item_name": "Fire rune",
                "tier": 1,
                "avg_auc": 0.90,
                "avg_positive_rate": 0.01,
                "recommended_offset": 0.015,
                "models": {
                    "1h_1.5pct": {
                        "path": "554/1h_1.5pct.cbm",
                        "auc": 0.95,
                        "brier": 0.004,
                        "calibrated": True,
                        "positive_rate": 0.008,
                    },
                    "1h_2.0pct": {
                        "path": "554/1h_2.0pct.cbm",
                        "auc": 0.96,
                        "brier": 0.003,
                        "calibrated": True,
                        "positive_rate": 0.005,
                    },
                },
            },
            "565": {
                "item_name": "Blood rune",
                "tier": 1,
                "avg_auc": 0.87,
                "avg_positive_rate": 0.011,
                "recommended_offset": 0.020,
                "models": {
                    "1h_2.0pct": {
                        "path": "565/1h_2.0pct.cbm",
                        "auc": 0.92,
                        "brier": 0.004,
                        "calibrated": True,
                        "positive_rate": 0.006,
                    }
                },
            },
        },
    }

    registry_path = tmp_path / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f)

    return registry_path


@pytest.fixture
def mock_db_connection():
    """Mock database connection string for testing."""
    return "postgresql://test:test@localhost:5432/test"
