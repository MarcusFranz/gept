# GePT Inference

PatchTST-based inference job for the GePT Grand Exchange prediction system.

## What's Here

- **Inference pipeline**: reads recent market data from Postgres/TimescaleDB and writes predictions back
- **Scripts**: backup and migration utilities

## Running Inference

```bash
python -m venv venv
source venv/bin/activate
cp .env.example .env
source .env
pip install -r requirements.txt
python src/pipeline/run_patchtst_inference.py --model-path /path/to/best_model.pt
```

The production job runs as a `systemd` timer that launches a one-shot Docker container (see `infra/systemd/gept-inference.*`).

## Local Setup

For full environment setup (database tunnel, env vars, dependencies), see `DEVELOPER_SETUP.md`.

## Deployment

The inference deployment unit is the Docker image `ghcr.io/marcusfranz/gept-inference`.

## Requirements

- Python 3.12+ (local dev)
- PostgreSQL with TimescaleDB
- PatchTST model checkpoint file (mounted into the container on the server)

## Environment Variables

`DB_PASS` is required for any database access; the rest have safe defaults.

```bash
export DB_PASS=replace-with-secure-password
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=osrs_data
export DB_USER=osrs_user
```

## License

Private - All rights reserved
