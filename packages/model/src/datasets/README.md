# GePT Dataset Management System

Clean, reproducible datasets for ML experiments. Extracts from PostgreSQL, computes features, and saves tensor-ready parquet files.

## Quick Start

### 1. List Available Recipes

```bash
python scripts/build_dataset.py --list
```

Output:
```
📋 Available Dataset Recipes:

  baseline
    Description: Standard production dataset: Tier 1-2 items, 6 months, 5-min granularity
    Granularity: 5min
    Feature Set: baseline

  high_volume
    Description: High-volume items (>1000 avg volume): most liquid, best for short-term predictions
    Granularity: 5min
    Feature Set: full

  ... (7 total recipes)
```

### 2. Build a Dataset from a Recipe

```bash
# Build baseline dataset
python scripts/build_dataset.py --recipe baseline

# Build with custom suffix
python scripts/build_dataset.py --recipe high_volume --suffix _jan2026
```

### 3. Load Dataset for Training

```python
from src.datasets.dataset_loader import DatasetLoader

loader = DatasetLoader()

# Load as numpy arrays (tensor-ready)
X_train, y_train = loader.load_numpy("baseline_1.0", split="train")
X_val, y_val = loader.load_numpy("baseline_1.0", split="val")
X_test, y_test = loader.load_numpy("baseline_1.0", split="test")

print(f"Train: X={X_train.shape}, y={y_train.shape}")
# Train: X=(50000, 120), y=(50000, 108)
```

## What Gets Built

When you build a dataset, the system:

1. **Queries PostgreSQL** - Extracts `price_data_5min` (or `price_data_1min`) for selected items
2. **Computes Features** - 100+ features (moving averages, volatility, spreads, RSI, etc.)
3. **Computes Targets** - 108 fill probability targets (offsets × time windows)
4. **Splits Temporally** - 70% train / 15% val / 15% test (no data leakage)
5. **Saves Parquet Files** - Compressed, tensor-ready data

### Output Structure

```
datasets/
  baseline_1.0/
    ├── metadata.json               # Dataset info, checksums, stats
    ├── item_metadata.json          # Per-item statistics
    ├── train_features.parquet      # Training features (X)
    ├── train_targets.parquet       # Training targets (y)
    ├── val_features.parquet        # Validation features
    ├── val_targets.parquet         # Validation targets
    ├── test_features.parquet       # Test features
    └── test_targets.parquet        # Test targets
```

### Parquet File Format

**Features (X):**
- Columns: `item_id`, `high`, `low`, `mid`, `spread`, `spread_pct`, `mid_ma_1h`, `mid_ma_4h`, ... (120 columns)
- Rows: Timestamped observations (one row per 5-minute interval per item)
- Data type: float32
- No normalization (you handle this in your training code)

**Targets (y):**
- Columns: `target_fill_0.01_4h`, `target_fill_0.01_8h`, ... (108 targets)
- Values: Binary (0 or 1) indicating if roundtrip fills
- Data type: float32

## Available Recipes

| Recipe | Description | Items | Granularity | History |
|--------|-------------|-------|-------------|---------|
| `baseline` | Standard production dataset | Tier 1-2, >100 vol | 5-min | 6 months |
| `high_volume` | High-liquidity items only | >1000 volume | 5-min | 3 months |
| `recent_1min` | High-frequency data | Tier 1-2 | 1-min | 30 days |
| `seasonal_winter` | Winter months only (Dec-Feb) | Tier 1-3 | 5-min | Multi-year |
| `equipment_only` | Equipment items (weapons, armor) | Tier 1-2 | 5-min | 6 months |
| `ablation_minimal` | Minimal features (price + volume) | Tier 1-2 | 5-min | 6 months |
| `long_history` | Maximum history for trend learning | 2+ year history | 5-min | 2 years |

## Creating Custom Recipes

### Option 1: Python API

```python
from src.datasets.dataset_recipe import DatasetRecipe, ItemFilter, TimeFilter, DataGranularity

recipe = DatasetRecipe(
    name="my_experiment",
    description="Custom dataset for XYZ experiment",
    version="1.0",
    granularity=DataGranularity.FIVE_MINUTE,
    item_filter=ItemFilter(
        min_tier=1,
        max_tier=2,
        min_completeness=0.90,
        min_avg_volume=500,
        include_item_ids=[2, 314, 1513]  # Specific items
    ),
    time_filter=TimeFilter(
        lookback_days=90  # Last 3 months
    )
)

# Save to library
from src.datasets.dataset_recipe import RecipeLibrary
library = RecipeLibrary()
library.save_recipe(recipe)

# Build dataset
from src.datasets.dataset_builder import DatasetBuilder
builder = DatasetBuilder()
metadata = builder.build_from_recipe(recipe)
```

### Option 2: Edit YAML

```bash
# Copy an existing recipe
cp configs/dataset_recipes/baseline.yaml configs/dataset_recipes/my_experiment.yaml

# Edit the YAML
vim configs/dataset_recipes/my_experiment.yaml
```

Example YAML:
```yaml
name: my_experiment
description: "My custom experiment dataset"
version: "1.0"
granularity: 5min

item_filter:
  min_tier: 1
  max_tier: 2
  min_completeness: 0.85
  min_avg_volume: 500
  include_item_ids: [2, 314, 1513]  # Optional: specific items

time_filter:
  lookback_days: 90  # Last 3 months
  # Or use absolute dates:
  # date_start: "2025-01-01"
  # date_end: "2025-04-01"

split_config:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

feature_set: baseline
output_format: parquet
compression: snappy
```

Then build:
```bash
python scripts/build_dataset.py --recipe my_experiment
```

## Loading Datasets

### Wide Format (Pandas)

```python
from src.datasets.dataset_loader import DatasetLoader

loader = DatasetLoader()
X_df, y_df = loader.load_wide_format("baseline_1.0", split="train")

# X_df is a DataFrame with columns: item_id, high, low, mid, spread, ...
# y_df is a DataFrame with columns: target_fill_0.01_4h, ...
```

### Numpy Arrays (Tensor-Ready)

```python
X_train, y_train = loader.load_numpy("baseline_1.0", split="train")
X_val, y_val = loader.load_numpy("baseline_1.0", split="val")

# X_train: (n_samples, n_features) - float32 numpy array
# y_train: (n_samples, n_targets) - float32 numpy array
```

### PyTorch Tensors

```python
X_train, y_train = loader.load_pytorch("baseline_1.0", split="train")

# X_train: torch.Tensor of shape (n_samples, n_features)
# y_train: torch.Tensor of shape (n_samples, n_targets)

# Use in PyTorch DataLoader
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

for X_batch, y_batch in dataloader:
    # Training loop
    pass
```

### Dataset Info

```python
# Print comprehensive dataset info
loader.print_dataset_info("baseline_1.0")

# Get metadata
metadata = loader.load_metadata("baseline_1.0")
print(metadata['item_count'])  # Number of items
print(metadata['train_rows'])  # Training rows

# Get feature/target names
features = loader.get_feature_names("baseline_1.0")
targets = loader.get_target_names("baseline_1.0")
```

## Example Training Workflow

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.datasets.dataset_loader import load_dataset

# 1. Load dataset
X_train, y_train = load_dataset("baseline_1.0", format="numpy", split="train")
X_val, y_val = load_dataset("baseline_1.0", format="numpy", split="val")

# 2. Normalize features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)

# 3. Train on first target (target_fill_0.02_24h)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_norm, y_train[:, 0])

# 4. Evaluate
val_score = model.score(X_val_norm, y_val[:, 0])
print(f"Validation accuracy: {val_score:.3f}")
```

## PyTorch + PatchTST Example

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.datasets.dataset_loader import DatasetLoader

# Load dataset
loader = DatasetLoader()
X_train, y_train = loader.load_pytorch("baseline_1.0", split="train")
X_val, y_val = loader.load_pytorch("baseline_1.0", split="val")

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Define model (simplified)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

# Train
model = SimpleModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Dataset Versions

Each dataset build creates a versioned directory (e.g., `baseline_1.0`). Versions are immutable:

- **Same recipe, different data** = new version suffix (`baseline_1.0_jan2026`)
- **Modified recipe** = increment version in recipe YAML
- **Checksums** ensure data integrity (validated on load)

## Tips

### For Quick Experiments
Use `recent_1min` or `high_volume` for fast iteration (smaller datasets).

### For Production Models
Use `baseline` or `long_history` for robust training.

### For Ablation Studies
Use `ablation_minimal` (minimal features) as baseline, then compare to `baseline` (full features).

### For Specific Item Types
Create custom recipes filtering by:
- Volume ranges (`min_avg_volume`, `max_avg_volume`)
- Specific item lists (`include_item_ids`)
- Data quality (`min_tier`, `min_completeness`)

### Memory Management
Large datasets (>1M rows) can use chunked loading:

```python
# Load in chunks for large datasets
X_df, y_df = loader.load_wide_format("long_history_1.0", split="train")
chunk_size = 100000

for i in range(0, len(X_df), chunk_size):
    X_chunk = X_df.iloc[i:i+chunk_size]
    y_chunk = y_df.iloc[i:i+chunk_size]
    # Process chunk
```

## Troubleshooting

### "Dataset not found"
```bash
# List available datasets
ls datasets/

# Or use Python
from src.datasets.dataset_loader import DatasetLoader
loader = DatasetLoader()
print(loader.dataset_dir)
```

### "Recipe not found"
```bash
# List recipes
python scripts/build_dataset.py --list

# Or initialize standard recipes
python scripts/build_dataset.py --init-recipes
```

### "Checksum mismatch"
Dataset files were modified after creation. Rebuild:
```bash
python scripts/build_dataset.py --recipe baseline
```

### Out of Memory
Reduce dataset size by:
- Using fewer items (`include_item_ids` in recipe)
- Shorter time window (`lookback_days`)
- Higher granularity (5-min instead of 1-min)

## File Locations

```
packages/model/
  ├── src/datasets/
  │   ├── dataset_builder.py     # Build datasets from PostgreSQL
  │   ├── dataset_loader.py      # Load datasets for training
  │   ├── dataset_recipe.py      # Recipe definitions
  │   └── README.md              # This file
  ├── configs/dataset_recipes/   # Recipe YAML files
  │   ├── baseline.yaml
  │   ├── high_volume.yaml
  │   └── ...
  ├── datasets/                  # Built datasets (gitignored)
  │   ├── baseline_1.0/
  │   ├── high_volume_1.0/
  │   └── ...
  └── scripts/
      └── build_dataset.py       # CLI for building datasets
```

## Next Steps

1. **Build your first dataset**: `python scripts/build_dataset.py --recipe baseline`
2. **Inspect it**: `python -m src.datasets.dataset_loader baseline_1.0`
3. **Load in training script**: See examples above
4. **Create custom recipes**: Edit YAML or use Python API
5. **Experiment**: Compare models on different datasets for reproducibility
