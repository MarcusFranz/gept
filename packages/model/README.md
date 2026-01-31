# GePT Model - Inference & Data Collection

Inference engine and data collectors for the GePT Grand Exchange prediction system.

## What's Here

- **Inference**: CatBoost multi-target models predicting fill probabilities for OSRS GE trades
- **Collectors**: Data ingestion services (5min, 1min, hourly prices + news + player counts)
- **Scripts**: Deployment, backup, and migration utilities

## Running Inference

```bash
# Dry run (no database writes)
python run_inference.py --dry-run

# Full inference cycle
python run_inference.py
```

Inference runs every 5 minutes via cron, generating predictions for 399 items across 108 targets.

## Data Collectors

See `collectors/` for Docker-based collection services. Deploy with:

```bash
./deploy_collectors.sh
```

## Deployment

```bash
# From repo root
./deploy-ampere.sh           # Full deploy
./deploy-ampere.sh --quick   # Skip dependency install
./deploy-ampere.sh --status  # Check current state
```

## Requirements

- Python 3.10+
- PostgreSQL with TimescaleDB
- CatBoost trained models in `models/` directory

## License

Private - All rights reserved
