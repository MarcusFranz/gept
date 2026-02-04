# GePT Model - Inference & Data Collection

Inference engine and data collectors for the GePT Grand Exchange prediction system.

## What's Here

- **Inference**: PatchTST quantile regression models predicting fill probabilities for OSRS GE trades
- **Collectors**: Data ingestion services (5min, 1min, hourly prices + news + player counts)
- **Scripts**: Deployment, backup, and migration utilities

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

## Environment

Most scripts expect the database credentials to be available as environment variables:
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASS`

For the full setup (SSH tunnel, `.env` workflow, and common troubleshooting), see
`packages/model/DEVELOPER_SETUP.md`.

## License

Private - All rights reserved
