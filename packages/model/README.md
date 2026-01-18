# GePT - Grand Exchange Prediction Tool

A machine learning system for predicting profitable trading opportunities on the Old School RuneScape (OSRS) Grand Exchange.

## Overview

GePT uses CatBoost gradient boosting models trained on 5 years of historical price data (426M+ rows) to predict the probability that buy/sell order pairs will fill within specific time windows.

### Key Features

- **314 item models** with 5,596 trained classifiers
- **Multi-horizon predictions**: 1h, 2h, 4h, 8h, 12h, 24h windows
- **Multiple profit margins**: 1.5%, 2.0%, 2.5% offsets
- **Real-time inference**: Predictions refresh every 5 minutes
- **Calibrated probabilities**: Models use AUC-optimized training with validation

## Project Structure

```
gept/
├── src/                    # Core library
│   ├── feature_engine.py   # 102-feature computation pipeline
│   ├── target_engine.py    # Fill probability target generation
│   ├── batch_predictor_fast.py  # Production inference engine
│   ├── trainer.py          # Model training pipeline
│   └── ...
├── training/               # Training scripts
│   ├── train_production_catboost.py  # Main training script
│   └── ...
├── scripts/                # Utility scripts
│   ├── run_inference_cron.sh  # Cron job wrapper
│   ├── setup_predictions_table.sql  # Database schema
│   └── ...
├── cloud/                  # Cloud deployment configs
├── tests/                  # Test suite
├── docs/                   # Analysis reports
├── run_inference.py        # Inference entry point
├── deploy_ampere.sh        # Deployment script
└── RECOMMENDATION_ENGINE.md  # Integration guide
```

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL with TimescaleDB
- Access to OSRS price data database

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/gept.git
cd gept

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Set up database connection in environment or config:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=osrs_data
export DB_USER=osrs_user
export DB_PASSWORD=your_password
```

2. Place SSH keys in `.secrets/` directory (not tracked by git)

### Running Inference

```bash
# Dry run (no database writes)
python run_inference.py --dry-run --models-dir models

# Full inference cycle
python run_inference.py --models-dir models
```

### Training Models

```bash
cd training
python train_production_catboost.py
```

## Model Architecture

### Features (102 total)

- **Spread features**: Current and moving average spreads
- **Price movement**: Returns at multiple lookback periods
- **Volatility**: Standard deviation and Parkinson volatility
- **Volume**: Moving averages and ratios
- **Time features**: Hour/day cyclical encoding, weekend flags
- **Momentum**: Price-volume divergence indicators
- **RSI**: Relative Strength Index at 4h and 24h

### Targets

Binary classification: Will a roundtrip trade (buy low, sell high) complete within X hours at Y% profit margin?

### Model

- **Algorithm**: CatBoost Classifier
- **Optimization**: AUC (ROC-AUC)
- **Validation**: Temporal train/val/test split

## Database Schema

See [scripts/setup_predictions_table.sql](scripts/setup_predictions_table.sql) for the full schema.

Key tables:
- `predictions`: Model outputs (refreshed every 5 min)
- `actual_fills`: Tracking for calibration monitoring
- `price_data_5min`: Historical price data

## Deployment

The system runs on an Ampere ARM server with:
- Cron job every 5 minutes
- ~2 minute inference cycle
- TimescaleDB for time-series storage

See [RECOMMENDATION_ENGINE.md](RECOMMENDATION_ENGINE.md) for integration details.

## Performance

| Metric | Value |
|--------|-------|
| Items covered | 314 |
| Models | 5,596 |
| Predictions/cycle | 5,596 |
| Cycle time | ~2 minutes |
| High confidence | 72% of predictions |

## License

Private - All rights reserved

## Acknowledgments

- OSRS Wiki for price data API
- RuneLite for market data collection
