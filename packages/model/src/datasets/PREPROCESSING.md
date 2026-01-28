# Data Preprocessing & Quality Control

Additional cleaning and validation beyond basic feature computation.

## What Gets Preprocessed

When you build a dataset with preprocessing enabled (default), the system automatically:

1. **Removes duplicates** - Identical feature rows
2. **Handles missing values** - Drops features with >10% missing, fills rest with 0
3. **Clips outliers** - Values beyond ±5 standard deviations
4. **Removes low-variance features** - Constant or near-constant columns
5. **Validates targets** - Drops imbalanced targets (<20 positive samples or <0.1% positive rate)
6. **Removes sparse rows** - Rows with >5% missing features

## Default Configuration

```python
PreprocessingConfig(
    # Sparsity filters
    max_missing_pct=0.05,           # Drop rows with >5% missing features
    max_feature_missing_pct=0.10,   # Drop features with >10% missing

    # Outlier handling
    clip_outliers=True,
    outlier_std_threshold=5.0,      # Clip beyond ±5 std

    # Feature validation
    drop_constant_features=True,
    min_feature_variance=1e-8,

    # Target validation
    min_positive_samples=20,        # Need ≥20 positive samples
    drop_imbalanced_targets=True,
    max_imbalance_ratio=0.001,      # Drop if <0.1% positive

    # Misc
    drop_duplicates=True,
    verbose=True
)
```

## Using Custom Preprocessing

### Option 1: Build with Custom Config

```python
from src.datasets import DatasetBuilder, PreprocessingConfig

config = PreprocessingConfig(
    max_missing_pct=0.10,          # More lenient (allow 10% missing)
    clip_outliers=False,            # Don't clip outliers
    drop_imbalanced_targets=False   # Keep all targets
)

builder = DatasetBuilder(
    enable_preprocessing=True,
    preprocessing_config=config
)

metadata = builder.build_from_recipe_name("baseline")
```

### Option 2: Disable Preprocessing

```python
# Build raw dataset (no preprocessing)
builder = DatasetBuilder(enable_preprocessing=False)
metadata = builder.build_from_recipe_name("baseline")

# Then apply custom preprocessing manually
from src.datasets.data_preprocessor import DataPreprocessor
from src.datasets.dataset_loader import DatasetLoader

loader = DatasetLoader()
X_train, y_train = loader.load_wide_format("baseline_1.0", "train")

preprocessor = DataPreprocessor(config)
X_clean, y_clean, stats = preprocessor.fit_transform(X_train, y_train)
```

## Preprocessing Output Files

After building a dataset with preprocessing, you get extra files:

```
datasets/baseline_1.0/
├── preprocessing_stats.json    # What got removed/clipped
├── quality_report.json         # Detailed quality metrics
├── ... (regular dataset files)
```

### preprocessing_stats.json

```json
{
  "initial_rows": 1500000,
  "initial_features": 125,
  "initial_targets": 110,
  "steps": [
    {
      "step": "remove_duplicates",
      "rows_removed": 1234,
      "duplicate_pct": 0.08
    },
    {
      "step": "handle_missing_features",
      "features_dropped": 3,
      "dropped_features": ["bad_feature_1", "bad_feature_2", "bad_feature_3"]
    },
    {
      "step": "clip_outliers",
      "features_clipped": 12,
      "total_values_clipped": 5432
    },
    {
      "step": "validate_targets",
      "targets_dropped": 5,
      "dropped_targets": ["target_fill_0.03_48h", ...]
    },
    {
      "step": "remove_sparse_rows",
      "rows_removed": 2345,
      "sparse_pct": 0.16
    }
  ],
  "final_rows": 1496421,
  "final_features": 122,
  "final_targets": 105,
  "rows_removed": 3579,
  "rows_removed_pct": 0.24
}
```

### quality_report.json

```json
{
  "feature_quality": {
    "high": {
      "mean": 123.45,
      "std": 45.67,
      "min": 10.0,
      "max": 500.0,
      "missing_pct": 0.0,
      "zeros_pct": 0.01,
      "unique_values": 45678
    },
    "spread_pct": {
      "mean": 0.025,
      "std": 0.015,
      ...
    },
    ... (all features)
  },
  "target_quality": {
    "target_fill_0.02_24h": {
      "positive_samples": 125000,
      "negative_samples": 1371421,
      "positive_rate": 0.0835,
      "imbalance_ratio": 0.0911
    },
    ... (all targets)
  },
  "overall": {
    "total_samples": 1496421,
    "total_features": 122,
    "total_targets": 105,
    "total_missing_values": 0,
    "missing_pct": 0.0
  }
}
```

## Preprocessing Summary Example

When building, you'll see:

```
======================================================================
DATA PREPROCESSING SUMMARY
======================================================================

Initial Dataset:
  Rows:     1,500,000
  Features: 125
  Targets:  110

Preprocessing Steps:

  Remove Duplicates:
    Rows removed: 1,234

  Handle Missing Features:
    Features dropped: 3
      ['bad_feature_1', 'bad_feature_2', 'bad_feature_3']

  Clip Outliers:
    Features clipped: 12
    Values clipped: 5,432

  Remove Low Variance Features:
    Features dropped: 0

  Validate Targets:
    Targets dropped: 5

  Remove Sparse Rows:
    Rows removed: 2,345

Final Dataset:
  Rows:     1,496,421 (0.2% removed)
  Features: 122
  Targets:  105
======================================================================
```

## Why Each Step Matters

### 1. Remove Duplicates
**Issue:** Exact same feature vectors (probably from data collection bugs)
**Impact:** Inflates dataset size, can bias models
**Solution:** Drop duplicates, keep first occurrence

### 2. Handle Missing Features
**Issue:** Some features have systematic missingness (e.g., RSI can't compute with <14 periods)
**Impact:** Models struggle with features that are mostly NaN
**Solution:** Drop features with >10% missing, fill rest with 0

### 3. Clip Outliers
**Issue:** Extreme values from low-volume trades or data errors (e.g., price spike from 100 to 10000)
**Impact:** Breaks normalization, dominates gradients
**Solution:** Clip to ±5 std (keeps 99.9999% of normal data, removes extremes)

### 4. Remove Low-Variance Features
**Issue:** Features that are constant (e.g., all zeros) provide no signal
**Impact:** Wasted computation, can cause numerical issues
**Solution:** Drop features with variance < 1e-8

### 5. Validate Targets
**Issue:** Some targets have too few positive samples (e.g., only 5 fills for a rare offset/window combo)
**Impact:** Model can't learn from <20 samples, evaluation is unreliable
**Solution:** Drop targets with <20 positive samples or <0.1% positive rate

### 6. Remove Sparse Rows
**Issue:** Some rows have too many missing features (e.g., first 300 rows where MAs haven't stabilized)
**Impact:** Poor training samples, model learns to handle NaN instead of patterns
**Solution:** Drop rows with >5% missing features

## When to Adjust Preprocessing

### More Aggressive Cleaning (Higher Quality, Less Data)

```python
PreprocessingConfig(
    max_missing_pct=0.01,           # Very strict (only 1% missing allowed)
    max_feature_missing_pct=0.05,   # Drop features with >5% missing
    outlier_std_threshold=3.0,      # Clip beyond ±3 std (more aggressive)
    min_positive_samples=50,        # Need more positive samples
    max_imbalance_ratio=0.005       # Only keep targets with ≥0.5% positive
)
```

Use for: Production models, final evaluation

### Less Aggressive (Lower Quality, More Data)

```python
PreprocessingConfig(
    max_missing_pct=0.10,           # Allow 10% missing
    max_feature_missing_pct=0.20,   # Keep features with up to 20% missing
    clip_outliers=False,            # Don't clip (let model handle)
    drop_imbalanced_targets=False,  # Keep all targets
    drop_constant_features=False    # Keep all features
)
```

Use for: Exploratory analysis, data-hungry models

### No Preprocessing (Raw Data)

```python
builder = DatasetBuilder(enable_preprocessing=False)
```

Use for: Custom preprocessing pipelines, debugging feature computation

## Inspecting Quality Reports

```python
import json

# Load quality report
with open("datasets/baseline_1.0/quality_report.json", 'r') as f:
    report = json.load(f)

# Check feature quality
for feature, stats in report['feature_quality'].items():
    if stats['missing_pct'] > 5:
        print(f"{feature}: {stats['missing_pct']:.2f}% missing")

# Check target balance
for target, stats in report['target_quality'].items():
    if stats['positive_rate'] < 0.05:
        print(f"{target}: only {stats['positive_rate']:.2%} positive")

# Overall stats
print(report['overall'])
```

## Best Practices

1. **Use default preprocessing for most experiments** - It's well-tuned for GePT data
2. **Inspect quality_report.json after building** - Know your data before training
3. **Keep preprocessing disabled for ablation studies** - So feature comparisons are fair
4. **Use aggressive preprocessing for production** - Higher quality > more data
5. **Check preprocessing_stats.json if results are unexpected** - Maybe too much got dropped

## Example: Building with Custom Preprocessing

```bash
# First, build raw dataset (no preprocessing)
python scripts/build_dataset.py --recipe baseline --no-preprocessing

# Then apply custom preprocessing in Python
```

```python
from src.datasets import DatasetLoader, DataPreprocessor, PreprocessingConfig

# Load raw dataset
loader = DatasetLoader()
X_train, y_train = loader.load_wide_format("baseline_1.0", "train")

# Custom preprocessing
config = PreprocessingConfig(
    clip_outliers=True,
    outlier_std_threshold=3.0,  # More aggressive clipping
    drop_imbalanced_targets=True,
    min_positive_samples=100     # Higher threshold
)

preprocessor = DataPreprocessor(config)
X_clean, y_clean, stats = preprocessor.fit_transform(X_train, y_train)

# Now train on cleaned data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_clean)

model = LogisticRegression()
model.fit(X_norm, y_clean.iloc[:, 0])  # First target
```

## Summary

Preprocessing is **enabled by default** and provides:
- ✅ Cleaner data
- ✅ Better model performance
- ✅ Fewer NaN handling headaches
- ✅ Quality reports for debugging

Disable it only if you need full control over cleaning logic.
