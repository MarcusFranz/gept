# GePT Inference

PatchTST-based inference job for the GePT Grand Exchange prediction system.

## What's Here

- **Inference pipeline**: reads recent market data from Postgres/TimescaleDB and writes predictions back
- **Scripts**: backup and migration utilities

## Running Inference

```bash
pip install -r requirements.txt
export DB_PASS=replace-with-secure-password
python src/pipeline/run_patchtst_inference.py --model-path /path/to/best_model.pt
```

## GePT3 Beta Overlay

Use these scripts to publish GePT3 checkpoint outputs under a beta `model_id`
without changing API contracts:

```bash
# 1) Export checkpoint -> artifact directory
python scripts/gept3/export_gept3_checkpoint_artifact.py \
  --checkpoint /path/to/epoch4.pt \
  --donor-dataset /path/to/gept3_hybrid_snap_...npz \
  --out-dir /data/gept/models/gept3/gept3_hybrid_e4_beta

# 2) Write beta prediction rows
python src/pipeline/run_gept3_beta_inference.py \
  --artifact-dir /data/gept/models/gept3/gept3_hybrid_e4_beta \
  --base-model-id gept_v2 \
  --out-model-id gept3_hybrid_e4_beta
```

Cron wrapper:

```bash
scripts/run_gept3_beta_inference_cron.sh
```

The production job runs as a `systemd` timer that launches a one-shot Docker container (see `infra/systemd/gept-inference.*`).

## Model Validation Loop

Validate prediction quality and recommendation quality against realized market
conditions, then persist steering signals for model iteration:

```bash
python src/validation/run_model_validation.py \
  --window-hours 168 \
  --max-rows 200000 \
  --output-json logs/model_validation_latest.json
```

Optional single-model run:

```bash
python src/validation/run_model_validation.py --model-id gept_v2
```

This pipeline writes to:

- `model_validation_prediction_outcomes`: row-level realized outcomes for each matured prediction
- `model_validation_snapshots`: per-model summary metrics plus development signals

If migrations are managed manually, apply:

```bash
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
  -f scripts/migrations/019_model_validation_system.sql
```

Cron wrapper:

```bash
scripts/run_model_validation_cron.sh
```

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
