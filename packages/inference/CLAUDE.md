# GePT Inference - Claude Code Instructions

## Project Overview

GePT is a machine learning system for OSRS Grand Exchange price prediction.
This package contains the **inference job** (not collectors, not the API).

Production inference runs on a server via a `systemd` timer that launches a one-shot Docker container.

## Package Structure

```
packages/inference/
├── src/
│   ├── db_utils.py                     # Database connections (DB_PASS required)
│   ├── feature_engine.py               # Feature computation
│   ├── target_engine.py                # Target/EV calculation
│   └── pipeline/
│       └── run_patchtst_inference.py   # PatchTST production inference
├── scripts/
│   ├── backup_database.sh              # DB backup helper
│   └── migrations/                     # SQL schema migrations
├── Dockerfile                          # GHCR image build
└── deploy_ampere.sh                    # Ops helper (server-side deployment)
```

Related deployment units:

- `packages/collectors/`: data collection + monitoring (Docker Compose)
- `packages/engine/`: recommendation API (Docker Compose)
- `packages/web/`: Vercel site

Server `systemd` units live in `infra/systemd/`.

## Database Access

- Required env var: `DB_PASS`
- Defaults: `DB_HOST=localhost`, `DB_PORT=5432`, `DB_NAME=osrs_data`, `DB_USER=osrs_user`

## Deployment Model

- Docker image: `ghcr.io/marcusfranz/gept-inference`
- Server unit: `infra/systemd/gept-inference.service` + `infra/systemd/gept-inference.timer`
- Environment file (server): `/home/ubuntu/gept/.env`
- Model checkpoint is mounted from the host into the container (see the systemd unit)

## Sensitive Files

- Never commit `.env` or SSH keys; keep secrets in `.secrets/` (git-ignored).
