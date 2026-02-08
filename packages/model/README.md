# GePT Model - Inference & Data Collection

Inference engine and data collectors for the GePT Grand Exchange prediction system.

## What's Here

- **Inference**: PatchTST quantile regression models predicting fill probabilities for OSRS GE trades
- **Collectors**: Data ingestion services (5min, 1min, hourly prices + news + player counts)
- **Scripts**: Deployment, backup, and migration utilities

## Running Inference

```bash
# From packages/model
# Dry run (no database writes)
python run_inference.py --dry-run

# Full inference cycle
python run_inference.py
```

Inference runs every 5 minutes via cron, generating predictions across 7 horizons and 5 quantiles per item.

## Local Setup

For full environment setup (database tunnel, env vars, dependencies), see `DEVELOPER_SETUP.md`.

## Data Collectors

See `collectors/` for Docker-based collection services. Deploy with:

```bash
# From packages/model
./deploy_collectors.sh
```

## Deployment

```bash
# From packages/model
./deploy-ampere.sh           # Full deploy
./deploy-ampere.sh --quick   # Skip dependency install
./deploy-ampere.sh --status  # Check current state
```

## Requirements

- Python 3.10+
- PostgreSQL with TimescaleDB
- PatchTST model checkpoint in `models/patchtst/` directory

## License

Private - All rights reserved
