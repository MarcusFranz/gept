"""
Centralized CatBoost parameter configuration.

Loads CatBoost hyperparameters from config/training_config.yaml to ensure
consistency across all training scripts.

Usage:
    from catboost_config import CATBOOST_PARAMS, load_catboost_params

    # Use default params
    model = CatBoostClassifier(**CATBOOST_PARAMS)

    # Or load with custom config path
    params = load_catboost_params('/path/to/config.yaml')
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_catboost_params(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load CatBoost parameters from training config YAML.

    Args:
        config_path: Path to config file. Defaults to config/training_config.yaml

    Returns:
        Dictionary of CatBoost hyperparameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If config file missing required sections
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    catboost_params = config["training"]["catboost"]

    # Convert task_type from string if needed (YAML stores as string)
    if "task_type" in catboost_params:
        catboost_params["task_type"] = str(catboost_params["task_type"])

    return catboost_params


# Default params for import convenience
# This loads once at module import time
CATBOOST_PARAMS = load_catboost_params()
