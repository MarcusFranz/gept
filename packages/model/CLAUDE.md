# GePT Model - Claude Code Instructions

## Project Overview

GePT is a machine learning system for OSRS Grand Exchange price prediction. This package contains **inference and data collection** code only. Training code has been archived locally (see below).

The system runs inference every 5 minutes on an Ampere server, generating fill probability predictions for 399 items across 108 targets (18 time windows × 6 price offsets).

## Package Structure

```
packages/model/
├── run_inference.py              # Inference entry point
├── src/
│   ├── batch_predictor_multitarget.py  # Production inference engine
│   ├── feature_engine.py         # Feature computation (102 features)
│   ├── target_engine.py          # Target/EV calculation
│   ├── db_utils.py               # Database connection pool
│   ├── inference_config.py       # Probability caps, confidence tiers
│   ├── calibration.py            # Model calibration manager
│   ├── catboost_config.py        # CatBoost model config
│   └── training_tier_config.py   # Time-window tier definitions
├── collectors/                   # Data collection services (Docker + systemd)
├── scripts/
│   ├── run_inference_cron.sh     # Cron wrapper
│   ├── run_inference_fallback.sh # Fallback inference
│   ├── run_inference_hydra.sh    # Hydra server inference
│   ├── backup_database.sh        # Daily DB backup
│   ├── check_collectors.sh       # Collector health check
│   ├── ssh_tunnel.sh             # DB tunnel helper
│   └── migrations/               # SQL schema migrations
├── deploy_ampere.sh              # Deploy to Ampere
├── deploy_collectors.sh          # Deploy collectors
└── rollback_ampere.sh            # Rollback deployment
```

## Training Code

Training code (CatBoost pipeline, Optuna tuning, cloud training, etc.) has been removed from the public repo and archived locally at `docs/archived-training/`. To restore for local work:

```bash
# From repo root
cp -r docs/archived-training/packages/model/src/pipeline packages/model/src/pipeline
cp -r docs/archived-training/packages/model/cloud packages/model/cloud
```

**Do NOT commit training code back to the public repo.**

## Database Access

PostgreSQL database with OSRS market data:

- **Connection**: `postgresql://osrs_user:$DB_PASS@localhost:5432/osrs_data`
- **SSH Tunnel**: `ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 $AMPERE_HOST`
  - `AMPERE_HOST` default: `ubuntu@150.136.170.128`

### Key Tables
- `price_data_5min` - 426M rows of 5-min prices (2021-2026)
- `predictions` - Model predictions (refreshed every 5 min)
- `items` - 4,500 item metadata

## Deployment

- **Host**: `ubuntu@150.136.170.128`
- **SSH Key**: `.secrets/oracle_key.pem`
- **Deploy inference**: `./deploy_ampere.sh` (from repo root)
- **Deploy collectors**: `./deploy_collectors.sh`

See `RECOMMENDATION_ENGINE.md` for full integration details.

## Data Collection Services

All data collection services are in the `collectors/` directory and run on the Ampere server.

### Docker Collectors (docker-compose)
| Service | Container | Port | Interval | Description |
|---------|-----------|------|----------|-------------|
| 5-Minute | osrs-ge-collector | 9100 | 5 min | OHLC prices for all items |
| Hourly | osrs-hourly-collector | 9101 | 1 hour | Historical hourly data |
| News | osrs-news-collector | 9102 | 30 min | Wiki changes, news |
| 1-Minute | osrs-latest-1m | 9103 | 1 min | High-frequency ticks |
| Dashboard | osrs-dashboard | 8080 | - | Status dashboard |

### Systemd Collectors
| Service | Port | Interval | Description |
|---------|------|----------|-------------|
| player_count.service | 9104 | 60 sec | OSRS player count |
| item_updater.service | 9105 | 24 hours | Item metadata sync |

### Collector Commands

```bash
./deploy_collectors.sh              # Deploy all collectors
./deploy_collectors.sh --quick      # Quick deploy (no rebuild)
./scripts/check_collectors.sh       # Check status
./scripts/check_collectors.sh --brief  # Brief status
```

## Database Utilities

Use `src/db_utils.py` for database connections:

```python
from db_utils import get_connection, release_connection, CONN_PARAMS

conn = get_connection()
try:
    with conn.cursor() as cur:
        cur.execute("SELECT ...")
finally:
    release_connection(conn)
```

## Sensitive Files

All sensitive files are in `.secrets/` (git-ignored):
- `oracle_key.pem` - Ampere server SSH key
- `vast_key.pem` - Vast.ai SSH key

## Git Workflow

**Never commit directly to main.** Always use feature branches and PRs.

### Branch naming:

| Prefix | Use for |
|--------|---------|
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `refactor/<name>` | Code cleanup |
| `chore/<name>` | Config, docs, dependencies |

### Commit message prefixes:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring |
| `chore:` | Maintenance tasks |
| `docs:` | Documentation |
