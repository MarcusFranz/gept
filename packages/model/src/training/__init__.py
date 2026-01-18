"""
GePT Training Pipeline Module
============================

Automated daily training pipeline with intelligent model lifecycle management.

Components:
    - item_selector: Intelligent item selection for training
    - remote_trainer: Remote GPU training orchestrator
    - model_validator: Model validation before deployment
    - lifecycle_manager: Model lifecycle state machine
    - performance_tracker: Performance drift detection

Usage:
    # Select items for training
    from src.training.item_selector import ItemSelector
    selector = ItemSelector()
    items = selector.select_items_for_training()

    # Validate and deploy models
    from src.training.lifecycle_manager import ModelLifecycleManager
    lifecycle = ModelLifecycleManager()
    lifecycle.activate_model(model_id)

See Issue #28 for full documentation.
"""

from .item_selector import ItemSelector
from .model_validator import ModelValidator
from .lifecycle_manager import ModelLifecycleManager

__all__ = [
    'ItemSelector',
    'ModelValidator',
    'ModelLifecycleManager',
]
