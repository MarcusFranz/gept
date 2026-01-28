# Dataset System Quick Start

## What You Have

A complete system to extract data from PostgreSQL → compute features → save clean parquet files → load as tensors for training.

## The 3-Step Workflow

### Step 1: Build a Dataset (extracts from PostgreSQL)

```bash
# List available dataset recipes
python scripts/build_dataset.py --list

# Build the baseline dataset
python scripts/build_dataset.py --recipe baseline
```

This will:
1. Query PostgreSQL `price_data_5min` table
2. Select Tier 1-2 items with good data quality
3. Compute 100+ features (moving averages, volatility, spreads, etc.)
4. Compute 108 targets (fill probability for different offsets/windows)
5. Split temporally (70% train / 15% val / 15% test)
6. Save to `datasets/baseline_1.0/` as parquet files

Output:
```
✅ Dataset Build Complete!

Version: baseline_1.0
Description: Standard production dataset: Tier 1-2 items, 6 months, 5-min granularity
Items: 287
Date Range: 2025-07-28 to 2026-01-27
Granularity: 5 minutes

Dataset Split:
  Train: 1,234,567 rows
  Val:   264,836 rows
  Test:  264,836 rows
  Total: 1,764,239 rows

Quality Metrics:
  Avg Completeness: 92.45%
  Avg Volume: 1,234

Files saved to:
  /Users/marcusfranz/Documents/gept/packages/model/datasets/baseline_1.0
```

### Step 2: Inspect the Dataset

```bash
python -m src.datasets.dataset_loader baseline_1.0
```

Output shows:
- Dataset metadata
- Feature names (120 features)
- Target names (108 targets)
- Array shapes for train/val/test

### Step 3: Load in Your Training Code

```python
from src.datasets.dataset_loader import load_dataset

# Load as numpy arrays (tensor-ready)
X_train, y_train = load_dataset("baseline_1.0", format="numpy", split="train")
X_val, y_val = load_dataset("baseline_1.0", format="numpy", split="val")
X_test, y_test = load_dataset("baseline_1.0", format="numpy", split="test")

print(X_train.shape)  # (1234567, 120) - 1.2M samples, 120 features
print(y_train.shape)  # (1234567, 108) - 108 binary targets

# Now train your model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)

model = LogisticRegression()
model.fit(X_train_norm, y_train[:, 0])  # Train on first target
```

## What's in the Parquet Files?

### Features (X) - `train_features.parquet`
Columns (120 total):
- `item_id` - Item identifier
- `high`, `low`, `mid` - Current prices
- `spread`, `spread_pct` - Bid-ask spread
- `mid_ma_1h`, `mid_ma_4h`, `mid_ma_8h`, ... - Moving averages
- `return_15min`, `return_1h`, `return_4h`, ... - Returns
- `volatility_1h`, `volatility_4h`, `volatility_24h` - Volatility measures
- `volume_ma_1h`, `volume_ma_4h`, ... - Volume features
- `rsi_14`, `rsi_24` - Momentum indicators
- And 90+ more...

### Targets (y) - `train_targets.parquet`
Columns (108 total):
- `target_fill_0.01_4h` - Does 1% offset fill within 4 hours?
- `target_fill_0.01_8h` - Does 1% offset fill within 8 hours?
- `target_fill_0.015_4h` - Does 1.5% offset fill within 4 hours?
- ... 108 combinations of (offset × time_window)

Each target is binary (0 or 1).

## Available Datasets to Build

| Recipe | Best For | Size | Build Time |
|--------|----------|------|------------|
| `baseline` | Standard experiments | Medium | ~5-10 min |
| `high_volume` | Active trading strategies | Small | ~2-3 min |
| `recent_1min` | High-frequency experiments | Medium | ~3-5 min |
| `long_history` | Long-term pattern learning | Large | ~15-20 min |
| `ablation_minimal` | Feature importance studies | Medium | ~5-10 min |

## Creating Your Own Dataset Mix

### Option 1: Use Existing Recipe with Modifications

```bash
# Copy an existing recipe
cp configs/dataset_recipes/baseline.yaml configs/dataset_recipes/my_dataset.yaml

# Edit to your needs
vim configs/dataset_recipes/my_dataset.yaml
```

Key things to modify:
```yaml
item_filter:
  min_avg_volume: 1000        # Only high-volume items
  include_item_ids: [2, 314]  # Specific items only

time_filter:
  lookback_days: 90           # Last 3 months only
```

### Option 2: Python API

```python
from src.datasets import DatasetRecipe, ItemFilter, TimeFilter, DataGranularity

recipe = DatasetRecipe(
    name="my_dataset",
    description="My custom dataset for experiments",
    granularity=DataGranularity.FIVE_MINUTE,
    item_filter=ItemFilter(
        min_tier=1,
        max_tier=2,
        min_avg_volume=500,
        include_item_ids=[2, 314, 1513]  # Cannonball, Feather, Magic logs
    ),
    time_filter=TimeFilter(
        lookback_days=60  # Last 2 months
    )
)

# Build it
from src.datasets import DatasetBuilder
builder = DatasetBuilder()
metadata = builder.build_from_recipe(recipe)
```

## Using 1-Minute Data

```bash
# Build 1-minute granularity dataset
python scripts/build_dataset.py --recipe recent_1min
```

This uses `price_data_1min` table (1-minute intervals instead of 5-minute).
- More data points per item (1440/day vs 288/day)
- Better for high-frequency strategies
- Larger file sizes

## PyTorch Integration

```python
from src.datasets.dataset_loader import DatasetLoader
import torch
from torch.utils.data import TensorDataset, DataLoader

loader = DatasetLoader()

# Load as PyTorch tensors
X_train, y_train = loader.load_pytorch("baseline_1.0", split="train")
X_val, y_val = loader.load_pytorch("baseline_1.0", split="val")

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Train
for epoch in range(10):
    for X_batch, y_batch in train_loader:
        # Your training loop
        pass
```

## Comparison Across Datasets

```python
from src.datasets.dataset_loader import DatasetLoader

loader = DatasetLoader()

# Build multiple datasets
# python scripts/build_dataset.py --recipe baseline
# python scripts/build_dataset.py --recipe high_volume

# Load and compare
X_baseline, y_baseline = loader.load_numpy("baseline_1.0", "train")
X_high_vol, y_high_vol = loader.load_numpy("high_volume_1.0", "train")

print(f"Baseline: {X_baseline.shape}")    # (1.2M, 120)
print(f"High Vol: {X_high_vol.shape}")    # (0.3M, 120) - fewer items, more liquid

# Train same model on both, compare performance
```

## What's Different from Current Training Code?

### Before (ad-hoc)
```python
# Every experiment re-queries DB, re-computes features
df = fetch_from_db(item_id, date_range)
X = compute_features(df)
y = compute_targets(df)
X_train, X_val = split(X)  # Different split each time
```

### After (dataset system)
```python
# Build once, reuse forever
# $ python scripts/build_dataset.py --recipe baseline

# Load instantly (no DB queries, no feature computation)
X_train, y_train = load_dataset("baseline_1.0", split="train")
# Same data every time, reproducible results
```

## Tips

1. **Start with `baseline`** - Good balance of data quality and size
2. **Use `high_volume` for quick iteration** - Smaller, faster experiments
3. **Use `recent_1min` for PatchTST** - High-frequency patterns
4. **Use `long_history` for production models** - Maximum training data
5. **Keep raw features** - Normalize in your training code (already done ✓)

## Files Created

After building `baseline`:

```
datasets/baseline_1.0/
├── metadata.json               # Dataset stats, checksums
├── item_metadata.json          # Per-item info
├── train_features.parquet      # X_train (1.2M rows × 120 cols)
├── train_targets.parquet       # y_train (1.2M rows × 108 cols)
├── val_features.parquet        # X_val (260K rows × 120 cols)
├── val_targets.parquet         # y_val
├── test_features.parquet       # X_test (260K rows × 120 cols)
└── test_targets.parquet        # y_test
```

Sizes: ~200-500 MB per dataset (compressed parquet)

## Next Steps

1. Build baseline dataset: `python scripts/build_dataset.py --recipe baseline`
2. Try loading it: See examples above
3. Train a model on it
4. Compare to your current training pipeline
5. Create custom recipes for your specific experiments

See [README.md](README.md) for full documentation.
