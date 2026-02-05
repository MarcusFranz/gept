# GePT Model - Inference & Data Collection

Inference engine and data collectors for the GePT Grand Exchange prediction system.

## What's Here

- **Inference**: PatchTST quantile regression models predicting fill probabilities for OSRS GE trades
- **Collectors**: Data ingestion services (5min, 1min, hourly prices + news + player counts)
- **Scripts**: Deployment, backup, and migration utilities

## Quick Start (Local)

```bash
cd packages/model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create and load your .env (see DEVELOPER_SETUP.md for required values)
source .env

# Dry run (no database writes)
python run_inference.py --dry-run
```

For full environment setup and SSH tunneling instructions, see `DEVELOPER_SETUP.md`.

## Running Inference

```bash
# Dry run (no database writes)
python run_inference.py --dry-run

# Full inference cycle
python run_inference.py
```

Inference runs every 5 minutes via cron, generating predictions across 7 horizons and 5 quantiles per item.

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
- PatchTST model checkpoint in `models/patchtst/` directory

## License

Private - All rights reserved
