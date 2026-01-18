"""Model runner for loading and executing CatBoost models."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class ModelRunner:
    """Loads and runs CatBoost models for fill probability prediction."""

    # Standard hour windows and offset percentages
    HOUR_WINDOWS = [1, 2, 4, 8, 12, 24, 48]
    OFFSET_PERCENTAGES = [0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250]

    def __init__(self, registry_path: str, models_base_path: Optional[str] = None):
        """Initialize model runner.

        Args:
            registry_path: Path to models/registry.json
            models_base_path: Base path for model files.
                If None, uses parent of registry_path.
        """
        self.registry_path = Path(registry_path)
        self.models_base_path = (
            Path(models_base_path) if models_base_path else self.registry_path.parent
        )
        self.registry = self._load_registry()
        self.models: dict[str, CatBoostClassifier] = {}
        self.scalers: dict[str, Any] = {}

    def _load_registry(self) -> dict:
        """Load the model registry JSON.

        Returns:
            Registry dictionary with model metadata
        """
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Registry not found at {self.registry_path}")
            return {"metadata": {}, "items": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in registry: {e}")
            return {"metadata": {}, "items": {}}

    def _get_model_key(self, item_id: int, hour: int, offset: float) -> str:
        """Generate cache key for a model configuration.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage

        Returns:
            Cache key string
        """
        offset_str = f"{offset * 100:.1f}pct".replace(".", "_")
        return f"{item_id}_{hour}h_{offset_str}"

    def _get_model_config_key(self, hour: int, offset: float) -> str:
        """Generate registry config key.

        Args:
            hour: Hour window
            offset: Offset percentage

        Returns:
            Config key as used in registry (e.g., "1h_2.0pct")
        """
        return f"{hour}h_{offset * 100:.1f}pct"

    def _load_model(
        self, item_id: int, hour: int, offset: float
    ) -> Optional[CatBoostClassifier]:
        """Load a CatBoost model from disk.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage

        Returns:
            Loaded CatBoostClassifier or None if not found
        """
        item_str = str(item_id)
        config_key = self._get_model_config_key(hour, offset)

        if item_str not in self.registry.get("items", {}):
            logger.warning(f"Item {item_id} not in registry")
            return None

        item_config = self.registry["items"][item_str]
        if config_key not in item_config.get("models", {}):
            logger.warning(f"Config {config_key} not found for item {item_id}")
            return None

        model_info = item_config["models"][config_key]
        model_path = self.models_base_path / model_info["path"]

        try:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None

    def _load_scaler(self, item_id: int, hour: int, offset: float) -> Optional[Any]:
        """Load a scaler from disk.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage

        Returns:
            Loaded scaler or None if not found
        """
        item_str = str(item_id)
        config_key = self._get_model_config_key(hour, offset)

        if item_str not in self.registry.get("items", {}):
            return None

        item_config = self.registry["items"][item_str]
        if config_key not in item_config.get("models", {}):
            return None

        model_info = item_config["models"][config_key]
        if "scaler_path" not in model_info:
            return None

        scaler_path = self.models_base_path / model_info["scaler_path"]

        try:
            return joblib.load(scaler_path)
        except Exception as e:
            logger.warning(f"Failed to load scaler {scaler_path}: {e}")
            return None

    def _get_or_load_model(
        self, item_id: int, hour: int, offset: float
    ) -> Optional[CatBoostClassifier]:
        """Get model from cache or load from disk.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage

        Returns:
            CatBoostClassifier or None
        """
        key = self._get_model_key(item_id, hour, offset)

        if key not in self.models:
            model = self._load_model(item_id, hour, offset)
            if model is not None:
                self.models[key] = model

        return self.models.get(key)

    def _get_or_load_scaler(
        self, item_id: int, hour: int, offset: float
    ) -> Optional[Any]:
        """Get scaler from cache or load from disk.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage

        Returns:
            Scaler or None
        """
        key = self._get_model_key(item_id, hour, offset)

        if key not in self.scalers:
            scaler = self._load_scaler(item_id, hour, offset)
            if scaler is not None:
                self.scalers[key] = scaler

        return self.scalers.get(key)

    def predict(
        self, item_id: int, hour: int, offset: float, features: np.ndarray
    ) -> Optional[float]:
        """Get fill probability for specific item/hour/offset.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage
            features: Feature vector

        Returns:
            Fill probability (0-1) or None if model unavailable
        """
        model = self._get_or_load_model(item_id, hour, offset)
        if model is None:
            return None

        # Apply scaler if available
        scaler = self._get_or_load_scaler(item_id, hour, offset)
        if scaler is not None:
            try:
                features = scaler.transform(features.reshape(1, -1)).flatten()
            except Exception as e:
                logger.warning(f"Scaler failed for item {item_id}: {e}")

        try:
            # Reshape for prediction
            features_2d = features.reshape(1, -1)
            proba = model.predict_proba(features_2d)
            # Return probability of positive class (fill)
            return float(proba[0][1])
        except Exception as e:
            logger.error(f"Prediction failed for item {item_id}: {e}")
            return None

    def predict_all_configs(self, item_id: int, features: np.ndarray) -> list[dict]:
        """Get predictions for all hour/offset combinations for an item.

        Args:
            item_id: OSRS item ID
            features: Feature vector

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for hour in self.HOUR_WINDOWS:
            for offset in self.OFFSET_PERCENTAGES:
                prob = self.predict(item_id, hour, offset, features)
                if prob is not None:
                    predictions.append(
                        {
                            "hour_window": hour,
                            "offset_pct": offset,
                            "fill_probability": prob,
                        }
                    )

        return predictions

    def get_model_metadata(
        self, item_id: int, hour: int, offset: float
    ) -> Optional[dict]:
        """Get AUC, training date, etc. for a specific model.

        Args:
            item_id: OSRS item ID
            hour: Hour window
            offset: Offset percentage

        Returns:
            Metadata dictionary or None
        """
        item_str = str(item_id)
        config_key = self._get_model_config_key(hour, offset)

        if item_str not in self.registry.get("items", {}):
            return None

        item_config = self.registry["items"][item_str]
        if config_key not in item_config.get("models", {}):
            return None

        model_info = item_config["models"][config_key]
        return {
            "auc": model_info.get("auc"),
            "brier": model_info.get("brier"),
            "calibrated": model_info.get("calibrated", False),
            "positive_rate": model_info.get("positive_rate"),
        }

    def get_item_metadata(self, item_id: int) -> Optional[dict]:
        """Get metadata for an item.

        Args:
            item_id: OSRS item ID

        Returns:
            Item metadata dictionary or None
        """
        item_str = str(item_id)
        if item_str not in self.registry.get("items", {}):
            return None

        item_config = self.registry["items"][item_str]
        return {
            "item_name": item_config.get("item_name"),
            "tier": item_config.get("tier"),
            "avg_auc": item_config.get("avg_auc"),
            "avg_positive_rate": item_config.get("avg_positive_rate"),
            "recommended_offset": item_config.get("recommended_offset"),
        }

    def get_supported_items(self) -> list[int]:
        """Get list of item IDs that have models.

        Returns:
            List of supported item IDs
        """
        return [int(item_id) for item_id in self.registry.get("items", {}).keys()]

    def get_registry_metadata(self) -> dict:
        """Get registry metadata.

        Returns:
            Registry metadata dictionary
        """
        return self.registry.get("metadata", {})

    def preload_models(self, item_ids: Optional[list[int]] = None):
        """Preload models into cache.

        Args:
            item_ids: Optional list of item IDs to preload. If None, loads all.
        """
        items_to_load = item_ids if item_ids else self.get_supported_items()
        loaded = 0

        for item_id in items_to_load:
            for hour in self.HOUR_WINDOWS:
                for offset in self.OFFSET_PERCENTAGES:
                    if self._get_or_load_model(item_id, hour, offset) is not None:
                        loaded += 1

        logger.info(f"Preloaded {loaded} models")

    def clear_cache(self):
        """Clear model and scaler caches."""
        self.models.clear()
        self.scalers.clear()

    def health_check(self) -> dict:
        """Check model runner health.

        Returns:
            Health status dictionary
        """
        try:
            num_items = len(self.get_supported_items())
            cached_models = len(self.models)
            metadata = self.get_registry_metadata()

            return {
                "status": "ok" if num_items > 0 else "error",
                "component": "model_runner",
                "supported_items": num_items,
                "cached_models": cached_models,
                "model_type": metadata.get("model_type"),
                "registry_created": metadata.get("created_at"),
            }
        except Exception as e:
            return {
                "status": "error",
                "component": "model_runner",
                "error": str(e),
            }
