"""
GePT Dataset Management

Clean, reproducible datasets for ML experiments.
"""

from .dataset_builder import DatasetBuilder, DatasetMetadata
from .dataset_loader import DatasetLoader, load_dataset
from .dataset_recipe import (
    DatasetRecipe,
    RecipeLibrary,
    ItemFilter,
    TimeFilter,
    SplitConfig,
    DataGranularity,
    FeatureSet,
    initialize_recipe_library
)

__all__ = [
    # Builder
    'DatasetBuilder',
    'DatasetMetadata',

    # Loader
    'DatasetLoader',
    'load_dataset',

    # Recipes
    'DatasetRecipe',
    'RecipeLibrary',
    'ItemFilter',
    'TimeFilter',
    'SplitConfig',
    'DataGranularity',
    'FeatureSet',
    'initialize_recipe_library'
]
