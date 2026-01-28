"""
Dataset Recipe System for GePT Model Experiments

Defines reusable configurations for building datasets with different:
- Item selection criteria
- Time periods
- Data quality thresholds
- Feature sets
- Data granularity (1min, 5min, hourly)
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum


class DataGranularity(str, Enum):
    """Data granularity options"""
    ONE_MINUTE = "1min"
    FIVE_MINUTE = "5min"
    HOURLY = "1h"


class FeatureSet(str, Enum):
    """Predefined feature sets for ablation studies"""
    MINIMAL = "minimal"      # Price + volume only
    BASELINE = "baseline"    # Standard production features
    FULL = "full"            # All available features
    EXPERIMENTAL = "experimental"  # Includes new/untested features


@dataclass
class ItemFilter:
    """Criteria for selecting items"""

    # Quality filters
    min_tier: int = 2
    max_tier: int = 3
    min_completeness: float = 0.80

    # Volume filters
    min_avg_volume: Optional[int] = None
    max_avg_volume: Optional[int] = None

    # Category filters (OSRS item categories)
    categories: Optional[List[str]] = None  # e.g., ["weapon", "armor", "potion"]

    # Explicit item lists
    include_item_ids: Optional[List[int]] = None
    exclude_item_ids: Optional[List[int]] = None

    # History requirements
    min_rows: int = 5000
    min_history_days: Optional[int] = None


@dataclass
class TimeFilter:
    """Time period selection"""

    # Absolute dates
    date_start: Optional[str] = None  # ISO format: "2023-01-01"
    date_end: Optional[str] = None

    # Relative dates (from latest available)
    lookback_days: Optional[int] = None  # e.g., 180 for last 6 months

    # Seasonal filters
    include_months: Optional[List[int]] = None  # e.g., [12, 1, 2] for winter
    include_days_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    include_hours: Optional[List[int]] = None  # 0-23


@dataclass
class SplitConfig:
    """Train/val/test split configuration"""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Advanced split options
    split_by_time: bool = True  # False = random split (not recommended for time series)
    shuffle_items: bool = False  # Shuffle items (but keep temporal order per item)


@dataclass
class DatasetRecipe:
    """Complete dataset configuration"""

    # Metadata
    name: str
    description: str
    version: str = "1.0"

    # Data source
    granularity: DataGranularity = DataGranularity.FIVE_MINUTE
    source_table: Optional[str] = None  # Auto-derived from granularity if None

    # Selection criteria
    item_filter: ItemFilter = field(default_factory=ItemFilter)
    time_filter: TimeFilter = field(default_factory=TimeFilter)
    split_config: SplitConfig = field(default_factory=SplitConfig)

    # Feature configuration
    feature_set: FeatureSet = FeatureSet.BASELINE
    custom_features: Optional[List[str]] = None  # Explicit feature list
    exclude_features: Optional[List[str]] = None  # Features to exclude

    # Target configuration
    target_offsets: Optional[List[float]] = None  # e.g., [0.01, 0.015, 0.02, 0.025, 0.03]
    target_windows_hours: Optional[List[int]] = None  # e.g., [4, 8, 12, 24, 48]

    # Output configuration
    output_format: str = "parquet"  # parquet, csv, hdf5
    compression: str = "snappy"  # snappy, gzip, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to strings for YAML serialization
        if isinstance(data.get('granularity'), DataGranularity):
            data['granularity'] = data['granularity'].value
        if isinstance(data.get('feature_set'), FeatureSet):
            data['feature_set'] = data['feature_set'].value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetRecipe':
        """Create recipe from dictionary"""
        # Handle nested dataclasses
        if 'item_filter' in data and isinstance(data['item_filter'], dict):
            data['item_filter'] = ItemFilter(**data['item_filter'])
        if 'time_filter' in data and isinstance(data['time_filter'], dict):
            data['time_filter'] = TimeFilter(**data['time_filter'])
        if 'split_config' in data and isinstance(data['split_config'], dict):
            data['split_config'] = SplitConfig(**data['split_config'])

        # Handle enums
        if 'granularity' in data and isinstance(data['granularity'], str):
            data['granularity'] = DataGranularity(data['granularity'])
        if 'feature_set' in data and isinstance(data['feature_set'], str):
            data['feature_set'] = FeatureSet(data['feature_set'])

        return cls(**data)

    def save(self, path: str):
        """Save recipe to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> 'DatasetRecipe':
        """Load recipe from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def get_source_table(self) -> str:
        """Get source table name based on granularity"""
        if self.source_table:
            return self.source_table

        table_map = {
            DataGranularity.ONE_MINUTE: "price_data_1min",
            DataGranularity.FIVE_MINUTE: "price_data_5min",
            DataGranularity.HOURLY: "price_data_5min"  # Aggregate from 5min
        }
        return table_map[self.granularity]

    def get_granularity_minutes(self) -> int:
        """Get granularity in minutes"""
        granularity_map = {
            DataGranularity.ONE_MINUTE: 1,
            DataGranularity.FIVE_MINUTE: 5,
            DataGranularity.HOURLY: 60
        }
        return granularity_map[self.granularity]


class RecipeLibrary:
    """Manage collection of dataset recipes"""

    def __init__(self, recipes_dir: Optional[str] = None):
        if recipes_dir is None:
            # Auto-detect: Use /workspace in Docker, else local path
            if Path("/workspace").exists():
                recipes_dir = "/workspace/configs/dataset_recipes"
            else:
                recipes_dir = str(Path(__file__).parent.parent.parent / "configs" / "dataset_recipes")
        self.recipes_dir = Path(recipes_dir)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)

    def list_recipes(self) -> List[str]:
        """List all available recipes"""
        return [f.stem for f in self.recipes_dir.glob("*.yaml")]

    def load_recipe(self, name: str) -> DatasetRecipe:
        """Load a recipe by name"""
        path = self.recipes_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {name}")
        return DatasetRecipe.load(str(path))

    def save_recipe(self, recipe: DatasetRecipe, overwrite: bool = False):
        """Save a recipe to the library"""
        path = self.recipes_dir / f"{recipe.name}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Recipe already exists: {recipe.name}. Use overwrite=True.")
        recipe.save(str(path))

    def delete_recipe(self, name: str):
        """Delete a recipe"""
        path = self.recipes_dir / f"{name}.yaml"
        if path.exists():
            path.unlink()

    def get_recipe_info(self, name: str) -> Dict[str, Any]:
        """Get recipe metadata without loading full recipe"""
        recipe = self.load_recipe(name)
        return {
            'name': recipe.name,
            'description': recipe.description,
            'version': recipe.version,
            'granularity': recipe.granularity.value,
            'feature_set': recipe.feature_set.value
        }


# Predefined recipes for common use cases
def create_baseline_recipe() -> DatasetRecipe:
    """Standard production dataset"""
    return DatasetRecipe(
        name="baseline",
        description="Standard production dataset: Tier 1-2 items, 6 months, 5-min granularity",
        version="1.0",
        granularity=DataGranularity.FIVE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=2,
            min_completeness=0.85,
            min_avg_volume=100,
            min_rows=10000
        ),
        time_filter=TimeFilter(
            lookback_days=180
        ),
        feature_set=FeatureSet.BASELINE
    )


def create_high_volume_recipe() -> DatasetRecipe:
    """High-volume, liquid items only"""
    return DatasetRecipe(
        name="high_volume",
        description="High-volume items (>1000 avg volume): most liquid, best for short-term predictions",
        version="1.0",
        granularity=DataGranularity.FIVE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=2,
            min_completeness=0.90,
            min_avg_volume=1000,
            min_rows=15000
        ),
        time_filter=TimeFilter(
            lookback_days=90  # Recent data more relevant for high-volume
        ),
        feature_set=FeatureSet.FULL
    )


def create_recent_1min_recipe() -> DatasetRecipe:
    """Recent data with 1-minute granularity"""
    return DatasetRecipe(
        name="recent_1min",
        description="Last 30 days, 1-minute data: high-frequency trading patterns",
        version="1.0",
        granularity=DataGranularity.ONE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=2,
            min_completeness=0.85,
            min_avg_volume=500,
            min_rows=20000  # 30 days * 1440 min/day = 43,200 rows
        ),
        time_filter=TimeFilter(
            lookback_days=30
        ),
        feature_set=FeatureSet.BASELINE
    )


def create_seasonal_recipe() -> DatasetRecipe:
    """Seasonal patterns (e.g., holiday events)"""
    return DatasetRecipe(
        name="seasonal_winter",
        description="Winter months (Dec-Feb) across multiple years: holiday event patterns",
        version="1.0",
        granularity=DataGranularity.FIVE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=3,
            min_completeness=0.80
        ),
        time_filter=TimeFilter(
            include_months=[12, 1, 2]  # Winter months
        ),
        feature_set=FeatureSet.BASELINE
    )


def create_equipment_recipe() -> DatasetRecipe:
    """Equipment items only (different trading patterns than consumables)"""
    return DatasetRecipe(
        name="equipment_only",
        description="Equipment items: different price dynamics than consumables",
        version="1.0",
        granularity=DataGranularity.FIVE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=2,
            min_completeness=0.85,
            categories=["weapon", "armor", "amulet", "ring"],
            min_rows=10000
        ),
        time_filter=TimeFilter(
            lookback_days=180
        ),
        feature_set=FeatureSet.FULL
    )


def create_ablation_minimal_recipe() -> DatasetRecipe:
    """Minimal features for ablation study"""
    return DatasetRecipe(
        name="ablation_minimal",
        description="Minimal feature set: price and volume only (ablation baseline)",
        version="1.0",
        granularity=DataGranularity.FIVE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=2,
            min_completeness=0.85,
            min_rows=10000
        ),
        time_filter=TimeFilter(
            lookback_days=180
        ),
        feature_set=FeatureSet.MINIMAL,
        custom_features=["high", "low", "mid", "spread", "spread_pct",
                        "high_price_volume", "low_price_volume"]
    )


def create_long_history_recipe() -> DatasetRecipe:
    """Maximum history for long-term pattern learning"""
    return DatasetRecipe(
        name="long_history",
        description="2+ years of data: long-term patterns and trend learning",
        version="1.0",
        granularity=DataGranularity.FIVE_MINUTE,
        item_filter=ItemFilter(
            min_tier=1,
            max_tier=2,
            min_completeness=0.80,
            min_history_days=730,  # 2 years
            min_rows=100000
        ),
        time_filter=TimeFilter(
            lookback_days=730
        ),
        feature_set=FeatureSet.BASELINE
    )


def initialize_recipe_library():
    """Create all standard recipes in the library"""
    library = RecipeLibrary()

    recipes = [
        create_baseline_recipe(),
        create_high_volume_recipe(),
        create_recent_1min_recipe(),
        create_seasonal_recipe(),
        create_equipment_recipe(),
        create_ablation_minimal_recipe(),
        create_long_history_recipe()
    ]

    for recipe in recipes:
        try:
            library.save_recipe(recipe, overwrite=True)
            print(f"✓ Created recipe: {recipe.name}")
        except Exception as e:
            print(f"✗ Failed to create {recipe.name}: {e}")

    print(f"\nRecipes saved to: {library.recipes_dir}")


if __name__ == "__main__":
    # Initialize standard recipe library
    initialize_recipe_library()

    # Example: Load and inspect a recipe
    library = RecipeLibrary()
    print("\nAvailable recipes:")
    for name in library.list_recipes():
        info = library.get_recipe_info(name)
        print(f"  - {name}: {info['description']}")
