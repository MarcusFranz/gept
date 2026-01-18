# GePT Infrastructure Reference

Complete server documentation for AI agents performing infrastructure tasks.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         HYDRA SERVER                            │
│  Primary training and inference                                 │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  Training   │  │  Inference  │                               │
│  │  (GPU)      │  │  (cron */5) │                               │
│  └─────────────┘  └──────┬──────┘                               │
│                          │ writes predictions                   │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AMPERE SERVER                             │
│  ubuntu@150.136.170.128                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ PostgreSQL  │  │ Rec Engine  │  │      Collectors         │  │
│  │ (port 5432) │  │ (port 8000) │  │  (Docker Compose)       │  │
│  │             │  │             │  │                         │  │
│  │ osrs_data   │◄─┤ Reads       │  │  5min, 1min, hourly,    │  │
│  │ predictions │  │ predictions │  │  news, players, items   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  Also: Fallback inference (cron */5, only if Hydra stale)       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         VERCEL                                  │
│  packages/web deployment (Astro/SolidJS)                        │
│  Edge network, automatic deployments from main branch           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          NEON                                   │
│  User database for packages/web                                 │
│  Serverless PostgreSQL, connection pooling                      │
└─────────────────────────────────────────────────────────────────┘
```

## Server Details

### Ampere Server (Oracle Cloud)

**Connection:**
```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128
```

**Services:**

| Service | Type | Port | Status Command |
|---------|------|------|----------------|
| PostgreSQL | systemd | 5432 | `sudo systemctl status postgresql` |
| Recommendation Engine | systemd | 8000 | `sudo systemctl status recommendation-engine` |
| Collectors | Docker | various | `docker compose ps` |
| Prometheus | Docker | 9090 | via Docker Compose |
| Grafana | Docker | 3001 | via Docker Compose |

**Key Paths:**
```
/opt/recommendation-engine/     # Engine deployment
/home/ubuntu/gept/              # Model repo (inference fallback)
/home/ubuntu/gept/logs/         # Inference logs
/home/ubuntu/gept/models/       # Trained models
/var/lib/postgresql/            # Database data
```

**Database:**
```
Host: localhost (from Ampere) or via SSH tunnel
Port: 5432
Database: osrs_data
User: osrs_user
Tables: price_data_5min (426M+ rows), predictions, items, model_registry
```

### Hydra Server (Training + Primary Inference)

**Primary Functions:**
- GPU training for CatBoost models
- Primary inference (cron every 5 minutes)
- Writes predictions to Ampere PostgreSQL via SSH tunnel

**Inference Schedule:**
```
*/5 * * * * /home/user/gept/scripts/run_inference_hydra.sh
```

**Failover Logic:**
- Hydra runs inference every 5 minutes
- Ampere checks prediction freshness
- If predictions > 10 minutes old, Ampere runs fallback inference

### Vercel (Web Frontend)

**Deployment:**
- Automatic on push to main branch
- Preview deployments for PRs
- Edge network (global CDN)

**Environment Variables:**
- `PREDICTION_API`: Recommendation engine URL
- `DATABASE_URL`: Neon connection string
- `BETTER_AUTH_SECRET`: Auth encryption key

### Neon (User Database)

**Purpose:** User data for packages/web
- User profiles and settings
- Active trades and history
- Feedback records

**Connection:** Via `@neondatabase/serverless` driver

## Service Dependencies

```
recommendation-engine.service
├── Requires: postgresql.service
├── Reads: predictions table
└── If PostgreSQL restarts → engine connections break (auto-reconnect)

collectors (Docker Compose)
├── Writes to: PostgreSQL (price_data_5min, etc.)
└── If PostgreSQL restarts → collectors retry automatically

inference (Hydra cron)
├── Reads: price_data_5min from Ampere (SSH tunnel)
├── Writes: predictions table on Ampere
└── If fails → Ampere fallback activates after 10 min
```

## DANGER ZONES

**NEVER do these without explicit coordination:**

1. **Restart PostgreSQL** - Breaks engine connections, interrupts collectors
2. **Modify predictions table schema** - Engine depends on exact columns
3. **Kill Hydra inference** - Must enable Ampere fallback first
4. **Change collector intervals** - Affects data quality for training
5. **Modify model registry** - Engine uses for model lifecycle

## Safe Operations

**Always safe:**
```bash
# View logs
sudo journalctl -u recommendation-engine -n 100 --no-pager
docker compose logs -f collector-5m

# Check status
sudo systemctl status recommendation-engine
docker compose ps

# Read-only database queries
psql -h localhost -U osrs_user -d osrs_data -c "SELECT COUNT(*) FROM predictions"

# Health checks
curl http://localhost:8000/api/v1/health
```

**Safe with caution:**
```bash
# Restart engine (stateless, auto-reconnects to DB)
sudo systemctl restart recommendation-engine

# Restart single collector
docker compose restart collector-5m
```

## Common Commands

### SSH Access
```bash
# Ampere
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# SSH tunnel for database access
ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128
```

### Service Management
```bash
# Check engine
sudo systemctl status recommendation-engine
sudo systemctl restart recommendation-engine
sudo journalctl -u recommendation-engine -n 100 --no-pager

# Check collectors
cd /home/ubuntu/gept/collectors
docker compose ps
docker compose logs -f --tail=50

# Check PostgreSQL
sudo systemctl status postgresql
psql -h localhost -U osrs_user -d osrs_data
```

### Deployment
```bash
# Engine deployment (from Ampere)
cd /opt/recommendation-engine
./deploy.sh

# Verify deployment
curl http://localhost:8000/api/v1/health
sudo journalctl -u recommendation-engine -n 20 --no-pager
```

### Database Queries
```bash
# Connect
psql -h localhost -U osrs_user -d osrs_data

# Check prediction freshness
SELECT MAX(time) FROM predictions;

# Check recent predictions count
SELECT COUNT(*) FROM predictions WHERE time > NOW() - INTERVAL '1 hour';

# Check collector health (recent price data)
SELECT COUNT(*) FROM price_data_5min WHERE time > NOW() - INTERVAL '1 hour';
```

### Log Locations
```
/home/ubuntu/gept/logs/inference.log      # Inference runs
/var/log/postgresql/                       # PostgreSQL logs
journalctl -u recommendation-engine        # Engine logs
docker compose logs                         # Collector logs
```

## Monitoring

**Grafana Dashboard:** http://150.136.170.128:3001 (internal only)
- Prediction freshness
- Collector health
- Database metrics

**Health Endpoints:**
- Engine: `http://localhost:8000/api/v1/health`
- Collectors: `http://localhost:8080/status` (dashboard)

## Runbook Index

See `/infra/runbooks/` for step-by-step procedures:
- `restart-engine.md` - Safe engine restart
- `deploy-engine.md` - Deploy new engine version
- `deploy-model.md` - Deploy new trained models
- `restart-postgres.md` - PostgreSQL restart (requires coordination)
- `check-health.md` - Full system health check
- `rollback-engine.md` - Revert to previous engine version

## Emergency Contacts

- **Prediction Staleness**: Check Hydra cron, verify Ampere fallback
- **Engine Down**: Check systemd status, restart if needed
- **Database Issues**: Check PostgreSQL status, connection pool
- **Collector Failures**: Check Docker Compose, restart specific collector
