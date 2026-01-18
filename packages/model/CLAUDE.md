# GePT Model - Claude Code Instructions

## Project Overview

GePT is a machine learning system for OSRS Grand Exchange price prediction. The system runs inference every 5 minutes on an Ampere server, generating fill probability predictions for 399 items across 108 targets (18 time windows × 6 price offsets).

## Key Files

- `src/batch_predictor_multitarget.py` - Production inference engine (108-target multi-output models)
- `src/batch_predictor_fast.py` - Legacy per-target inference engine (use `--legacy` flag)
- `src/feature_engine.py` - Feature computation
- `src/target_engine.py` - Target generation
- `run_inference.py` - Inference entry point
- `cloud/train_runpod_multitarget.py` - GPU model training

## Database Access

PostgreSQL database with OSRS market data:

- **Connection**: `postgresql://osrs_user:$DB_PASS@localhost:5432/osrs_data`
- **SSH Tunnel**: `ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 $AMPERE_HOST`
  - `AMPERE_HOST` is defined in `config/servers.env` (default: `ubuntu@150.136.170.128`)

### Key Tables
- `price_data_5min` - 426M rows of 5-min prices (2021-2026)
- `predictions` - Model predictions (refreshed every 5 min)
- `items` - 4,500 item metadata
- `model_registry` - Model lifecycle tracking (PENDING → ACTIVE → DEPRECATED → SUNSET → ARCHIVED)
- `training_jobs` - Training pipeline execution history
- `model_performance` - Model drift detection and calibration metrics

### Example Query
```sql
SELECT item_id, timestamp, avg_high_price
FROM price_data_5min
WHERE item_id = 2
ORDER BY timestamp DESC
LIMIT 10;
```

## Deployment

The prediction engine runs on the Ampere server:
- **Host**: `$AMPERE_HOST` (default: `ubuntu@150.136.170.128`, configured in `config/servers.env`)
- **SSH Key**: `$AMPERE_SSH_KEY` (default: `.secrets/oracle_key.pem`)
- **Deploy inference**: `./deploy_ampere.sh`
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

### Monitoring Stack
| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| Prometheus | gept-prometheus | 9090 | Metrics collection |
| Grafana | gept-grafana | 3001 | Dashboards & visualization |

### Collector Commands

```bash
# Deploy all collectors
./deploy_collectors.sh

# Quick deploy (no rebuild)
./deploy_collectors.sh --quick

# Deploy monitoring only
./deploy_collectors.sh --monitoring

# Check status
./scripts/check_collectors.sh

# Brief status
./scripts/check_collectors.sh --brief
```

### Access URLs
Access URLs use `$AMPERE_IP` (default: `150.136.170.128`, configured in `config/servers.env`):
- **Dashboard**: `http://$AMPERE_IP:8080`
- **Prometheus**: `http://$AMPERE_IP:9090`
- **Grafana**: `http://$AMPERE_IP:3001` (admin / $GRAFANA_PASSWORD)

### Key Tables (Collectors)
- `price_data_5min` - 5-minute OHLC prices (TimescaleDB, compressed)
- `prices_latest_1m` - 1-minute tick data (high-frequency ticks)
- `player_counts` - Historical player counts (198k+ records)
- `items` - Item metadata (4,500 items)

### Database Utilities

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
- `wsl_key.pem` - WSL training machine SSH key
- `training.env` - Training pipeline environment variables
- `gept-*.json` - GCP service account

## Git Workflow

**Never commit directly to main.** Always use feature branches and PRs.

### Before ANY code changes:

```bash
git checkout main && git pull
git checkout -b feature/<name>   # or fix/, refactor/, chore/
```

### Branch naming:

| Prefix | Use for |
|--------|---------|
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `fix/issue-<number>-<name>` | Fixing a specific issue |
| `refactor/<name>` | Code cleanup |
| `chore/<name>` | Config, docs, dependencies |

### After making changes:

```bash
git add <files>
git commit -m "feat: description

Co-Authored-By: Claude <agent-name> <noreply@anthropic.com>"
git push -u origin <branch-name>
```

### Create PR:

```bash
gh pr create --title "feat: description" --body "$(cat <<'EOF'
## Summary
- Changes made

## Test plan
- [ ] Verification steps

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### For bugs, create issues:

```bash
gh issue create --title "Bug: description" --body "..."
```

### Commit message prefixes:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring |
| `chore:` | Maintenance tasks |
| `docs:` | Documentation |
| `test:` | Adding tests |

## Parallel Claude Instances

**IMPORTANT**: Multiple Claude instances may work on different issues simultaneously. All Claudes share the same local folder, so you MUST use git worktrees to avoid conflicts.

### Always use a separate worktree for your work:

```bash
# Get repo info
REPO_ROOT=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$REPO_ROOT")
REPO_PARENT=$(dirname "$REPO_ROOT")

# Create a worktree for your feature branch (in parent directory)
git fetch origin
git worktree add "$REPO_PARENT/${REPO_NAME}-issue-XX" -b feat/your-feature origin/master

# Work in the isolated worktree
cd "$REPO_PARENT/${REPO_NAME}-issue-XX"
# ... make changes, commit, push, create PR ...

# Clean up when done (after PR is merged)
cd "$REPO_ROOT"
git worktree remove "$REPO_PARENT/${REPO_NAME}-issue-XX"
```

### Why worktrees:

- Each Claude gets an isolated working directory
- No conflicts with other Claudes' uncommitted changes
- Each worktree has its own branch checked out
- Changes in one worktree don't affect others

**Never edit files in the main repository folder** - always create a worktree first.

## /issue Workflow (MUST FOLLOW ALL PHASES)

When working on GitHub issues via `/issue <number>`, you MUST complete ALL phases:

### Phase 1-4: Setup, Analyze, Implement, Create PR
1. Create isolated worktree: `git worktree add "$REPO_PARENT/${REPO_NAME}-issue-XX" -b fix/issue-XX origin/master`
2. Fetch and analyze issue requirements
3. Enter plan mode, get user approval
4. Implement changes, run local tests
5. Commit and push, create PR with `Closes #XX`

### Phase 5: Monitor CI Checks (DO NOT SKIP)

After creating the PR, you MUST monitor CI until all checks pass:

```bash
# Wait for checks to start
sleep 15

# Check status
gh pr checks <pr-number>
```

If checks fail:
1. Get failure details: `gh run view <run-id> --log-failed`
2. Fix the issues locally
3. Commit and push the fix
4. Repeat until all checks pass (max 5 iterations)

### Phase 6: Verify Merge-Ready Status (DO NOT SKIP)

```bash
# Check for conflicts
git fetch origin master
gh pr view <pr-number> --json mergeable,mergeStateStatus
```

Required for merge-ready:
- All checks: `SUCCESS` or `SKIPPED`
- mergeable: `MERGEABLE`
- mergeStateStatus: `CLEAN`

### Phase 7: Final Output (REQUIRED)

Always end with one of these outputs:

**Success:**
```
============================================
PR Ready to Merge
============================================
PR: #<number>
Branch: <branch-name>
All CI checks passed.
No merge conflicts.
PR URL: <url>
============================================
```

**Blocked:**
```
============================================
PR NOT Ready - Manual Intervention Required
============================================
Blockers: <list issues>
============================================
```
