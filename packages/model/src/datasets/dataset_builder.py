"""
Clean Dataset Builder for GePT Model Experiments

Creates versioned, validated train/val/test datasets with:
- Temporal splits (no leakage)
- Precomputed features
- Quality validation
- Metadata tracking
"""

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from loguru import logger

from db_utils import get_simple_connection
from feature_engine import FeatureEngine, Granularity
from target_engine import TargetEngine
from .dataset_recipe import DatasetRecipe, RecipeLibrary
from .data_preprocessor import DataPreprocessor, PreprocessingConfig, generate_quality_report


@dataclass
class DatasetMetadata:
    """Metadata for a dataset version"""
    version: str
    created_at: str
    description: str

    # Data selection
    item_ids: List[int]
    item_count: int
    date_range_start: str
    date_range_end: str
    min_tier: int  # Minimum data quality tier

    # Split configuration
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_rows: int
    val_rows: int
    test_rows: int

    # Feature configuration
    feature_count: int
    target_count: int
    granularity_minutes: int

    # Data quality
    completeness_threshold: float
    avg_completeness: float
    avg_volume: float

    # File paths (relative to dataset root)
    train_features_path: str
    train_targets_path: str
    val_features_path: str
    val_targets_path: str
    test_features_path: str
    test_targets_path: str
    item_metadata_path: str

    # Checksums for validation
    train_checksum: str
    val_checksum: str
    test_checksum: str


class DatasetBuilder:
    """Build clean, versioned datasets for model experiments"""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        granularity_minutes: int = 5,
        enable_preprocessing: bool = True,
        preprocessing_config: Optional[PreprocessingConfig] = None
    ):
        # Auto-detect output directory based on environment
        if output_dir is None:
            if Path("/workspace").exists():
                output_dir = "/workspace/datasets"
            else:
                output_dir = str(Path(__file__).parent.parent.parent / "datasets")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.granularity = granularity_minutes
        self.enable_preprocessing = enable_preprocessing

        # Convert minutes to Granularity enum for FeatureEngine
        granularity_enum_map = {
            1: Granularity.ONE_MIN,
            5: Granularity.FIVE_MIN,
            60: Granularity.HOURLY
        }
        granularity_enum = granularity_enum_map.get(granularity_minutes, Granularity.FIVE_MIN)

        # Convert minutes to string for TargetEngine
        granularity_str_map = {
            1: '1m',
            5: '5m',
            60: '1h'
        }
        granularity_str = granularity_str_map.get(granularity_minutes, '5m')

        # Initialize engines
        self.feature_engine = FeatureEngine(granularity=granularity_enum)
        self.target_engine = TargetEngine(granularity=granularity_str)

        # Initialize preprocessor
        if self.enable_preprocessing:
            self.preprocessor = DataPreprocessor(preprocessing_config or PreprocessingConfig())
        else:
            self.preprocessor = None

        logger.info(f"DatasetBuilder initialized: {output_dir}")

    def build_from_recipe(
        self,
        recipe: DatasetRecipe,
        version_suffix: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Build a dataset from a recipe configuration

        Args:
            recipe: DatasetRecipe defining all configuration
            version_suffix: Optional suffix to add to recipe name (e.g., "_run1")

        Returns:
            DatasetMetadata with paths and statistics
        """
        # Generate version name
        version = f"{recipe.name}_{recipe.version}"
        if version_suffix:
            version = f"{version}{version_suffix}"

        logger.info(f"Building dataset from recipe: {recipe.name}")

        # Apply recipe item filters
        item_ids = self._apply_recipe_item_filter(recipe.item_filter)

        # Apply recipe time filters
        date_start, date_end = self._apply_recipe_time_filter(recipe.time_filter)

        # Check for unimplemented recipe features
        if recipe.custom_features:
            logger.warning(f"custom_features specified but not yet implemented: {recipe.custom_features}")
        if recipe.exclude_features:
            logger.warning(f"exclude_features specified but not yet implemented: {recipe.exclude_features}")
        if recipe.target_offsets:
            logger.warning(f"target_offsets specified but not yet implemented: {recipe.target_offsets}")
        if recipe.target_windows_hours:
            logger.warning(f"target_windows_hours specified but not yet implemented: {recipe.target_windows_hours}")

        # Build dataset using standard method
        return self.build_dataset(
            version=version,
            description=recipe.description,
            item_ids=item_ids,
            min_tier=recipe.item_filter.min_tier,
            date_start=date_start,
            date_end=date_end,
            train_ratio=recipe.split_config.train_ratio,
            val_ratio=recipe.split_config.val_ratio,
            test_ratio=recipe.split_config.test_ratio,
            min_rows_per_item=recipe.item_filter.min_rows,
            completeness_threshold=recipe.item_filter.min_completeness
        )

    def build_from_recipe_name(
        self,
        recipe_name: str,
        version_suffix: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Build a dataset from a recipe name in the library

        Args:
            recipe_name: Name of recipe in the library
            version_suffix: Optional suffix to add to version

        Returns:
            DatasetMetadata with paths and statistics
        """
        library = RecipeLibrary()
        recipe = library.load_recipe(recipe_name)
        return self.build_from_recipe(recipe, version_suffix)

    def build_dataset(
        self,
        version: str,
        description: str,
        item_ids: Optional[List[int]] = None,
        min_tier: int = 1,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_rows_per_item: int = 5000,
        completeness_threshold: float = 0.80
    ) -> DatasetMetadata:
        """
        Build a complete dataset with train/val/test splits

        Args:
            version: Dataset version identifier (e.g., "v1.0", "2026-01-27")
            description: Human-readable description
            item_ids: Specific items to include (if None, auto-select by tier)
            min_tier: Minimum data quality tier (1=premium, 2=good, 3=fair)
            date_start: Start date (ISO format, e.g., "2023-01-01")
            date_end: End date (ISO format, defaults to latest available)
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            min_rows_per_item: Minimum rows required per item
            completeness_threshold: Minimum data completeness (0-1)

        Returns:
            DatasetMetadata with paths and statistics
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        logger.info(f"Building dataset version: {version}")
        logger.info(f"Description: {description}")

        # Create version directory
        version_dir = self.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Select items
        if item_ids is None:
            item_ids = self._select_items_by_tier(min_tier, completeness_threshold)

        logger.info(f"Selected {len(item_ids)} items for dataset")

        # Step 2: Determine date range
        if date_end is None:
            date_end = self._get_latest_date()

        if date_start is None:
            # Default: 6 months of history
            date_start = (pd.to_datetime(date_end) - timedelta(days=180)).strftime("%Y-%m-%d")

        logger.info(f"Date range: {date_start} to {date_end}")

        # Step 3: Build datasets per item, then concatenate
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        item_metadata = []

        # Use parallel processing if multiple items
        if len(item_ids) > 1:
            logger.info(f"Processing {len(item_ids)} items in parallel...")
            with ProcessPoolExecutor(max_workers=min(len(item_ids), os.cpu_count() or 4)) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self._build_item_dataset,
                        item_id=item_id,
                        date_start=date_start,
                        date_end=date_end,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        min_rows=min_rows_per_item
                    ): item_id
                    for item_id in item_ids
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        train_X, train_y, val_X, val_y, test_X, test_y, meta = future.result()

                        if train_X is None:
                            logger.warning(f"Skipping item {item_id}: insufficient data")
                            continue

                        # Add item_id column for tracking
                        train_X['item_id'] = item_id
                        val_X['item_id'] = item_id
                        test_X['item_id'] = item_id

                        all_train_X.append(train_X)
                        all_train_y.append(train_y)
                        all_val_X.append(val_X)
                        all_val_y.append(val_y)
                        all_test_X.append(test_X)
                        all_test_y.append(test_y)
                        item_metadata.append(meta)

                        logger.info(f"✓ Completed item {item_id}")

                    except Exception as e:
                        logger.error(f"Failed to process item {item_id}: {e}")
                        continue
        else:
            # Single item: process sequentially (no overhead)
            for idx, item_id in enumerate(item_ids):
                logger.info(f"Processing item {item_id} ({idx+1}/{len(item_ids)})")

                try:
                    # Load and preprocess
                    train_X, train_y, val_X, val_y, test_X, test_y, meta = self._build_item_dataset(
                        item_id=item_id,
                        date_start=date_start,
                        date_end=date_end,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        min_rows=min_rows_per_item
                    )

                    if train_X is None:
                        logger.warning(f"Skipping item {item_id}: insufficient data")
                        continue

                    # Add item_id column for tracking
                    train_X['item_id'] = item_id
                    val_X['item_id'] = item_id
                    test_X['item_id'] = item_id

                    all_train_X.append(train_X)
                    all_train_y.append(train_y)
                    all_val_X.append(val_X)
                    all_val_y.append(val_y)
                    all_test_X.append(test_X)
                    all_test_y.append(test_y)
                    item_metadata.append(meta)

                except Exception as e:
                    logger.error(f"Failed to process item {item_id}: {e}")
                    continue

        # Check if any items were successfully processed
        if not all_train_X:
            raise ValueError("No items successfully processed. Check item filters and data availability.")

        # Concatenate all items
        logger.info("Concatenating datasets...")
        train_X = pd.concat(all_train_X, ignore_index=True)
        train_y = pd.concat(all_train_y, ignore_index=True)
        val_X = pd.concat(all_val_X, ignore_index=True)
        val_y = pd.concat(all_val_y, ignore_index=True)
        test_X = pd.concat(all_test_X, ignore_index=True)
        test_y = pd.concat(all_test_y, ignore_index=True)

        logger.info(f"Train: {len(train_X)} rows, Val: {len(val_X)} rows, Test: {len(test_X)} rows")

        # Step 4: Apply preprocessing (if enabled)
        if self.enable_preprocessing and self.preprocessor:
            logger.info("Applying data preprocessing...")

            # Fit on train, transform on val/test (prevents data leakage)
            train_X, train_y, train_stats = self.preprocessor.fit_transform(train_X, train_y, item_metadata)
            val_X, val_y = self.preprocessor.transform(val_X, val_y)
            test_X, test_y = self.preprocessor.transform(test_X, test_y)

            # Save preprocessing stats (convert numpy types to Python types for JSON)
            preprocessing_stats_path = version_dir / "preprocessing_stats.json"
            with open(preprocessing_stats_path, 'w') as f:
                json.dump(train_stats, f, indent=2, default=int)

            # Generate quality report
            quality_report_path = version_dir / "quality_report.json"
            generate_quality_report(train_X, train_y, str(quality_report_path))

            logger.info(f"After preprocessing - Train: {len(train_X)} rows, Val: {len(val_X)} rows, Test: {len(test_X)} rows")

        # Step 5: Save datasets
        train_X_path = version_dir / "train_features.parquet"
        train_y_path = version_dir / "train_targets.parquet"
        val_X_path = version_dir / "val_features.parquet"
        val_y_path = version_dir / "val_targets.parquet"
        test_X_path = version_dir / "test_features.parquet"
        test_y_path = version_dir / "test_targets.parquet"
        item_meta_path = version_dir / "item_metadata.json"

        train_X.to_parquet(train_X_path, compression='snappy', index=False)
        train_y.to_parquet(train_y_path, compression='snappy', index=False)
        val_X.to_parquet(val_X_path, compression='snappy', index=False)
        val_y.to_parquet(val_y_path, compression='snappy', index=False)
        test_X.to_parquet(test_X_path, compression='snappy', index=False)
        test_y.to_parquet(test_y_path, compression='snappy', index=False)

        # Save item metadata
        with open(item_meta_path, 'w') as f:
            json.dump(item_metadata, f, indent=2)

        # Step 5: Compute checksums
        train_checksum = self._compute_checksum(train_X_path)
        val_checksum = self._compute_checksum(val_X_path)
        test_checksum = self._compute_checksum(test_X_path)

        # Step 6: Compute quality metrics
        avg_completeness = np.mean([m['completeness'] for m in item_metadata])
        avg_volume = np.mean([m['avg_volume'] for m in item_metadata])

        # Step 7: Create metadata
        metadata = DatasetMetadata(
            version=version,
            created_at=datetime.now().isoformat(),
            description=description,
            item_ids=[m['item_id'] for m in item_metadata],
            item_count=len(item_metadata),
            date_range_start=date_start,
            date_range_end=date_end,
            min_tier=min_tier,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            train_rows=len(train_X),
            val_rows=len(val_X),
            test_rows=len(test_X),
            feature_count=len(train_X.columns) - 1,  # Exclude item_id
            target_count=len(train_y.columns),
            granularity_minutes=self.granularity,
            completeness_threshold=completeness_threshold,
            avg_completeness=avg_completeness,
            avg_volume=avg_volume,
            train_features_path=str(train_X_path.relative_to(self.output_dir)),
            train_targets_path=str(train_y_path.relative_to(self.output_dir)),
            val_features_path=str(val_X_path.relative_to(self.output_dir)),
            val_targets_path=str(val_y_path.relative_to(self.output_dir)),
            test_features_path=str(test_X_path.relative_to(self.output_dir)),
            test_targets_path=str(test_y_path.relative_to(self.output_dir)),
            item_metadata_path=str(item_meta_path.relative_to(self.output_dir)),
            train_checksum=train_checksum,
            val_checksum=val_checksum,
            test_checksum=test_checksum
        )

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        logger.info(f"Dataset saved to: {version_dir}")
        logger.info(f"Metadata: {metadata_path}")

        return metadata

    def _build_item_dataset(
        self,
        item_id: int,
        date_start: str,
        date_end: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        min_rows: int
    ) -> Tuple[Optional[pd.DataFrame], ...]:
        """Build train/val/test for a single item"""

        # Determine source table based on granularity
        table_map = {
            1: "price_data_1min",
            5: "price_data_5min",
            60: "price_data_5min"  # Aggregate from 5min for hourly
        }
        source_table = table_map.get(self.granularity, "price_data_5min")

        # Load raw price data
        conn = get_simple_connection()
        try:
            query = f"""
                SELECT timestamp, avg_high_price, avg_low_price,
                       high_price_volume, low_price_volume
                FROM {source_table}
                WHERE item_id = %s
                  AND timestamp >= %s::timestamp
                  AND timestamp < %s::timestamp + interval '1 day'
                ORDER BY timestamp ASC
            """
            df = pd.read_sql(query, conn, params=(item_id, date_start, date_end))
        finally:
            conn.close()

        if len(df) < min_rows:
            return None, None, None, None, None, None, None

        # Compute features (df already has 'timestamp' column from SQL query)
        df = self.feature_engine.compute_features(df)

        # Remove duplicate columns (FeatureEngine may create duplicates)
        df = df.loc[:, ~df.columns.duplicated()]

        # Compute targets
        df = self.target_engine.compute_targets(df)

        # Drop warmup rows (features stabilizing) and horizon rows (targets need future)
        warmup_rows = 300
        max_window = 48 * (60 // self.granularity)  # 48 hours in periods
        df = df.iloc[warmup_rows:-max_window].copy()

        if len(df) < min_rows:
            return None, None, None, None, None, None, None

        # Separate features and targets
        target_cols = [col for col in df.columns if 'target_' in col or 'fills_' in col]
        # Exclude both 'time' and 'timestamp' from features (timestamp was used for time features, now redundant)
        feature_cols = [col for col in df.columns if col not in target_cols and col not in ['time', 'timestamp']]

        X = df[feature_cols].copy()
        y = df[target_cols].copy()

        # Handle NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Temporal split
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_X = X.iloc[:train_end]
        train_y = y.iloc[:train_end]
        val_X = X.iloc[train_end:val_end]
        val_y = y.iloc[train_end:val_end]
        test_X = X.iloc[val_end:]
        test_y = y.iloc[val_end:]

        # Metadata
        metadata = {
            'item_id': item_id,
            'total_rows': n,
            'train_rows': len(train_X),
            'val_rows': len(val_X),
            'test_rows': len(test_X),
            'date_start': df['timestamp'].min().isoformat(),
            'date_end': df['timestamp'].max().isoformat(),
            'completeness': len(df) / self._expected_rows(date_start, date_end),
            'avg_volume': df[['high_price_volume', 'low_price_volume']].mean().mean()
        }

        return train_X, train_y, val_X, val_y, test_X, test_y, metadata

    def _apply_recipe_item_filter(self, item_filter) -> Optional[List[int]]:
        """Apply recipe item filter to select items"""
        from data_quality_analyzer import run_full_analysis

        # If explicit item IDs provided, use those
        if item_filter.include_item_ids:
            return item_filter.include_item_ids

        # Otherwise, select by criteria
        analysis = run_full_analysis()
        results = analysis['all_items']

        selected = []
        for item in results:
            # Tier filter
            if item['tier'] < item_filter.min_tier or item['tier'] > item_filter.max_tier:
                continue

            # Completeness filter
            if item['completeness'] < item_filter.min_completeness:
                continue

            # Volume filters
            if item_filter.min_avg_volume and item['avg_volume'] < item_filter.min_avg_volume:
                continue
            if item_filter.max_avg_volume and item['avg_volume'] > item_filter.max_avg_volume:
                continue

            # Exclude list
            if item_filter.exclude_item_ids and item['item_id'] in item_filter.exclude_item_ids:
                continue

            # TODO: Category filtering (requires item metadata table with categories)
            # if item_filter.categories:
            #     if item['category'] not in item_filter.categories:
            #         continue

            selected.append(item['item_id'])

        return selected

    def _apply_recipe_time_filter(self, time_filter) -> Tuple[str, str]:
        """Apply recipe time filter to determine date range

        Note: Currently only date_start, date_end, and lookback_days are implemented.
        TODO: Implement seasonal filters:
            - include_months: Filter by specific months
            - include_days_of_week: Filter by days of week
            - include_hours: Filter by hours of day
        """

        # Get latest available date
        latest_date = self._get_latest_date()

        # If explicit dates provided, use those
        if time_filter.date_start and time_filter.date_end:
            # TODO: Apply seasonal filters if specified
            if time_filter.include_months or time_filter.include_days_of_week or time_filter.include_hours:
                logger.warning("Seasonal filters (include_months, include_days_of_week, include_hours) not yet implemented")
            return time_filter.date_start, time_filter.date_end

        # If lookback specified, calculate start date
        if time_filter.lookback_days:
            end_date = time_filter.date_end or latest_date
            start_date = (pd.to_datetime(end_date) - timedelta(days=time_filter.lookback_days)).strftime("%Y-%m-%d")
            # TODO: Apply seasonal filters if specified
            if time_filter.include_months or time_filter.include_days_of_week or time_filter.include_hours:
                logger.warning("Seasonal filters (include_months, include_days_of_week, include_hours) not yet implemented")
            return start_date, end_date

        # Default: 6 months
        end_date = latest_date
        start_date = (pd.to_datetime(end_date) - timedelta(days=180)).strftime("%Y-%m-%d")
        return start_date, end_date

    def _select_items_by_tier(self, min_tier: int, completeness_threshold: float) -> List[int]:
        """Select items meeting tier and completeness criteria"""
        from data_quality_analyzer import run_full_analysis

        analysis = run_full_analysis()
        results = analysis['all_items']

        selected = []
        for item in results:
            if item['tier'] <= min_tier and item['completeness'] >= completeness_threshold:
                selected.append(item['item_id'])

        return selected

    def _get_latest_date(self) -> str:
        """Get the most recent date in price_data_5min"""
        conn = get_simple_connection()
        try:
            query = "SELECT MAX(timestamp) FROM price_data_5min"
            result = pd.read_sql(query, conn)
            return result.iloc[0, 0].strftime("%Y-%m-%d")
        finally:
            conn.close()

    def _expected_rows(self, date_start: str, date_end: str) -> int:
        """Calculate expected rows for date range"""
        start = pd.to_datetime(date_start)
        end = pd.to_datetime(date_end)
        days = (end - start).days
        if days <= 0:
            logger.warning(f"Invalid date range: {date_start} to {date_end}, days={days}")
            return 1  # Avoid division by zero in completeness calculation
        periods_per_day = 24 * (60 // self.granularity)  # 288 for 5-min
        return days * periods_per_day

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_dataset(self, version: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                    pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                    DatasetMetadata]:
        """Load a saved dataset by version"""
        version_dir = self.output_dir / version
        metadata_path = version_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset version not found: {version}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)

        metadata = DatasetMetadata(**metadata_dict)

        # Validate checksums
        logger.info("Validating checksums...")
        train_X_path = self.output_dir / metadata.train_features_path
        val_X_path = self.output_dir / metadata.val_features_path
        test_X_path = self.output_dir / metadata.test_features_path

        assert self._compute_checksum(train_X_path) == metadata.train_checksum, \
            "Train data checksum mismatch!"
        assert self._compute_checksum(val_X_path) == metadata.val_checksum, \
            "Val data checksum mismatch!"
        assert self._compute_checksum(test_X_path) == metadata.test_checksum, \
            "Test data checksum mismatch!"

        # Load datasets
        logger.info(f"Loading dataset version: {version}")
        train_X = pd.read_parquet(train_X_path)
        train_y = pd.read_parquet(self.output_dir / metadata.train_targets_path)
        val_X = pd.read_parquet(val_X_path)
        val_y = pd.read_parquet(self.output_dir / metadata.val_targets_path)
        test_X = pd.read_parquet(test_X_path)
        test_y = pd.read_parquet(self.output_dir / metadata.test_targets_path)

        logger.info(f"Loaded: Train={len(train_X)}, Val={len(val_X)}, Test={len(test_X)}")

        return train_X, train_y, val_X, val_y, test_X, test_y, metadata


if __name__ == "__main__":
    # Example usage
    builder = DatasetBuilder()

    # Build a clean dataset
    metadata = builder.build_dataset(
        version="v1.0-baseline",
        description="Baseline dataset: Tier 1-2 items, 6 months, 5-min granularity",
        min_tier=2,
        min_rows_per_item=10000,
        completeness_threshold=0.85
    )

    print(f"Dataset created: {metadata.version}")
    print(f"Items: {metadata.item_count}")
    print(f"Total rows: {metadata.train_rows + metadata.val_rows + metadata.test_rows}")
