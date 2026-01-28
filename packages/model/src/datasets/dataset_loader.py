"""
Dataset Loader for GePT Experiments

Loads preprocessed parquet datasets and converts to tensors for training.
Supports both wide format (pandas) and tensor format (numpy/torch).
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger


class DatasetLoader:
    """Load preprocessed datasets for training"""

    def __init__(self, dataset_dir: Optional[str] = None):
        # Auto-detect dataset directory based on environment
        if dataset_dir is None:
            if Path("/workspace").exists():
                dataset_dir = "/workspace/datasets"
            else:
                dataset_dir = str(Path(__file__).parent.parent.parent / "datasets")

        self.dataset_dir = Path(dataset_dir)

    def load_wide_format(
        self,
        version: str,
        split: str = "train"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset in wide pandas format

        Args:
            version: Dataset version name (e.g., "baseline_1.0")
            split: Which split to load ("train", "val", or "test")

        Returns:
            (X_df, y_df): Feature and target DataFrames
        """
        version_dir = self.dataset_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {version}")

        # Load metadata to get paths
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Determine paths
        if split == "train":
            X_path = self.dataset_dir / metadata['train_features_path']
            y_path = self.dataset_dir / metadata['train_targets_path']
        elif split == "val":
            X_path = self.dataset_dir / metadata['val_features_path']
            y_path = self.dataset_dir / metadata['val_targets_path']
        elif split == "test":
            X_path = self.dataset_dir / metadata['test_features_path']
            y_path = self.dataset_dir / metadata['test_targets_path']
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Load parquet files
        logger.info(f"Loading {split} split from {version}")
        X_df = pd.read_parquet(X_path)
        y_df = pd.read_parquet(y_path)

        logger.info(f"Loaded {len(X_df)} rows, {len(X_df.columns)} features, {len(y_df.columns)} targets")

        return X_df, y_df

    def load_numpy(
        self,
        version: str,
        split: str = "train",
        exclude_item_id: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset as numpy arrays (tensor-ready)

        Args:
            version: Dataset version name
            split: Which split to load
            exclude_item_id: Remove item_id from features (it's not a training feature)

        Returns:
            (X, y): Numpy arrays ready for model training
        """
        X_df, y_df = self.load_wide_format(version, split)

        # Remove item_id if present (not a feature for training)
        if exclude_item_id and 'item_id' in X_df.columns:
            X_df = X_df.drop(columns=['item_id'])

        X = X_df.to_numpy(dtype=np.float32)
        y = y_df.to_numpy(dtype=np.float32)

        return X, y

    def load_pytorch(
        self,
        version: str,
        split: str = "train",
        exclude_item_id: bool = True
    ):
        """
        Load dataset as PyTorch tensors

        Args:
            version: Dataset version name
            split: Which split to load
            exclude_item_id: Remove item_id from features

        Returns:
            (X_tensor, y_tensor): PyTorch tensors
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        X, y = self.load_numpy(version, split, exclude_item_id)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        return X_tensor, y_tensor

    def load_metadata(self, version: str) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_path = self.dataset_dir / version / "metadata.json"
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def load_item_metadata(self, version: str) -> list:
        """Load per-item metadata"""
        version_dir = self.dataset_dir / version
        metadata = self.load_metadata(version)
        item_meta_path = self.dataset_dir / metadata['item_metadata_path']

        with open(item_meta_path, 'r') as f:
            return json.load(f)

    def get_feature_names(self, version: str) -> list:
        """Get list of feature column names"""
        X_df, _ = self.load_wide_format(version, "train")
        return [col for col in X_df.columns if col != 'item_id']

    def get_target_names(self, version: str) -> list:
        """Get list of target column names"""
        _, y_df = self.load_wide_format(version, "train")
        return list(y_df.columns)

    def print_dataset_info(self, version: str):
        """Print comprehensive dataset information"""
        metadata = self.load_metadata(version)
        item_metadata = self.load_item_metadata(version)

        print(f"\n📊 Dataset: {version}")
        print(f"─" * 60)
        print(f"Description: {metadata['description']}")
        print(f"Created: {metadata['created_at']}")
        print(f"\n📈 Data Statistics:")
        print(f"  Items: {metadata['item_count']}")
        print(f"  Date Range: {metadata['date_range_start']} to {metadata['date_range_end']}")
        print(f"  Granularity: {metadata['granularity_minutes']} minutes")
        print(f"\n🎯 Split Information:")
        print(f"  Train: {metadata['train_rows']:,} rows ({metadata['train_ratio']:.0%})")
        print(f"  Val:   {metadata['val_rows']:,} rows ({metadata['val_ratio']:.0%})")
        print(f"  Test:  {metadata['test_rows']:,} rows ({metadata['test_ratio']:.0%})")
        print(f"  Total: {metadata['train_rows'] + metadata['val_rows'] + metadata['test_rows']:,} rows")
        print(f"\n🔢 Dimensions:")
        print(f"  Features: {metadata['feature_count']}")
        print(f"  Targets:  {metadata['target_count']}")
        print(f"\n✅ Quality Metrics:")
        print(f"  Avg Completeness: {metadata['avg_completeness']:.2%}")
        print(f"  Avg Volume: {metadata['avg_volume']:,.0f}")
        print(f"  Min Tier: {metadata['min_tier']}")
        print(f"\n📁 Files:")
        print(f"  Location: {self.dataset_dir / version}")
        print(f"  Train Features: {metadata['train_features_path']}")
        print(f"  Train Targets: {metadata['train_targets_path']}")


# Convenience function for quick loading
def load_dataset(
    version: str,
    format: str = "numpy",
    split: str = "train"
):
    """
    Quick load function

    Args:
        version: Dataset version name
        format: "wide", "numpy", or "pytorch"
        split: "train", "val", or "test"

    Returns:
        (X, y) in requested format
    """
    loader = DatasetLoader()

    if format == "wide":
        return loader.load_wide_format(version, split)
    elif format == "numpy":
        return loader.load_numpy(version, split)
    elif format == "pytorch":
        return loader.load_pytorch(version, split)
    else:
        raise ValueError(f"Invalid format: {format}. Must be 'wide', 'numpy', or 'pytorch'")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_loader.py <dataset_version>")
        print("Example: python dataset_loader.py baseline_1.0")
        sys.exit(1)

    version = sys.argv[1]
    loader = DatasetLoader()

    # Print info
    loader.print_dataset_info(version)

    # Show feature and target names
    print(f"\n📝 Features ({len(loader.get_feature_names(version))}):")
    features = loader.get_feature_names(version)
    for i in range(0, min(10, len(features))):
        print(f"  {features[i]}")
    if len(features) > 10:
        print(f"  ... and {len(features) - 10} more")

    print(f"\n🎯 Targets ({len(loader.get_target_names(version))}):")
    targets = loader.get_target_names(version)
    for i in range(0, min(10, len(targets))):
        print(f"  {targets[i]}")
    if len(targets) > 10:
        print(f"  ... and {len(targets) - 10} more")

    # Load and show shapes
    print(f"\n🔢 Array Shapes:")
    X_train, y_train = loader.load_numpy(version, "train")
    X_val, y_val = loader.load_numpy(version, "val")
    X_test, y_test = loader.load_numpy(version, "test")

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    print(f"\n✅ Dataset ready for training!")
