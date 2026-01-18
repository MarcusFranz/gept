# Data Collection Services Migration - Implementation Prompt

## Context

This is a 4-week project to migrate, optimize, and modernize the GePT data collection infrastructure running on an Ampere A1 server. All planning and analysis has been completed.

## Key Documentation References

**Read these first to understand the project**:

1. **[docs/DATA_COLLECTION_ANALYSIS.md](docs/DATA_COLLECTION_ANALYSIS.md)** - Complete analysis of all 8 data collection services currently running on the server, including resource usage, architecture, and recommendations.

2. **[collectors/README.md](collectors/README.md)** - Comprehensive operational guide covering service details, database schema, configuration, monitoring, and troubleshooting.

3. **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Detailed 4-week implementation plan with all phases, tasks, scripts, and verification steps.

4. **[CLAUDE.md](CLAUDE.md)** - Project overview with database access, deployment info, and sensitive file locations.

## Project Overview

**Current State**:
- 8 data collection services running on Ampere server (ubuntu@150.136.170.128)
- PostgreSQL/TimescaleDB database with 426M rows of price data
- Docker-based collectors + some systemd services
- Disk usage: 79GB/98GB (85% - approaching capacity)
- Mixed storage: PostgreSQL + DuckDB
- Hardcoded credentials across 9+ files

**Goal**:
- Consolidate all services to PostgreSQL (eliminate DuckDB)
- Enable TimescaleDB compression to reduce disk usage by 50-90%
- Refactor code to eliminate hardcoded credentials
- Set up Prometheus + Grafana monitoring
- Create deployment automation
- All while maintaining zero downtime

## Implementation Phases

### Week 1: Foundation & Safety
1. Stop redundant `osrs_collector_pg.py` process (PID 835141)
2. Fix dashboard health check in docker-compose.yml
3. Enable TimescaleDB compression on `price_data_5min` table
4. Set up automated backup strategy
5. Create `src/db_utils.py` and refactor 2-3 files

### Week 2: Service Consolidation
1. Migrate player count collector from DuckDB to PostgreSQL
2. Migrate item metadata updater to PostgreSQL
3. Create shared modules (`collectors/shared/metrics.py`, `collectors/shared/config.py`)

### Week 3: Monitoring
1. Deploy Prometheus server (Docker container)
2. Deploy Grafana and create dashboards
3. Set up alerting rules
4. Test and document monitoring stack

### Week 4: Automation & Polish
1. Create `deploy_collectors.sh` deployment script
2. Create `scripts/check_collectors.sh` status script
3. Comprehensive end-to-end testing
4. Update all documentation
5. Final verification

## Technical Details

**Server Access**:
- Host: `ubuntu@150.136.170.128`
- SSH Key: `.secrets/oracle_key.pem`
- Database: PostgreSQL on localhost:5432
- Database: `osrs_data`, User: `osrs_user`, Password: `$DB_PASS`

**Critical Files Already Created**:
- All collector scripts copied to `collectors/` (24 files)
- `.env.example` for environment configuration
- `.gitignore` for proper version control
- Comprehensive documentation in `docs/` and `collectors/`

**Safety Requirements**:
- **DO NOT interrupt data collection** - all changes must be non-disruptive
- Test all database changes on a backup first
- Run new and old services in parallel during migrations
- Monitor for 1 hour after each major change
- Have rollback plans ready

## Implementation Instructions

### For Week 1 Tasks:

**Task 1.1: Stop Redundant Collector**
```bash
# Connect to server and stop the redundant process
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128 'kill 835141'

# Verify it's not a systemd service
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128 'systemctl disable osrs_collector 2>&1 || echo "Not a systemd service"'

# Verify Docker containers still healthy
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128 'docker ps'

# Check data is still flowing
ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128 -N &
psql -h localhost -U osrs_user -d osrs_data -c "SELECT MAX(timestamp) FROM price_data_5min"
```

**Task 1.2: Fix Dashboard Health Check**
- Edit `collectors/docker-compose.yml`
- Change dashboard health check from `/metrics` to `/health`
- Deploy updated config
- Verify container shows "healthy"

**Task 1.3: Enable TimescaleDB Compression**
- Create `scripts/enable_timescaledb_compression.sql` (script provided in plan)
- Test on a copy of the data if possible
- Execute via SSH tunnel
- Verify compression stats
- Monitor query performance

**Task 1.4: Set Up Backups**
- Create `scripts/backup_database.sh` (script provided in plan)
- Test backup and restore
- Add to cron: `0 2 * * * /home/ubuntu/gept/scripts/backup_database.sh >> /home/ubuntu/gept/logs/backup.log 2>&1`

**Task 1.5: Create Database Utility Module**
- Create `src/db_utils.py` (code provided in plan)
- Refactor `src/batch_predictor_fast.py` to use it
- Refactor `src/monitoring.py` to use it
- Test inference: `python run_inference.py --dry-run`
- Test monitoring: `python src/monitoring.py --check all`

### For Week 2 Tasks:

**Task 2.1: Migrate Player Count Collector**
- Create PostgreSQL table (schema in plan)
- Create `scripts/migrate_player_counts.py` migration script
- Run migration on server
- Update collector to `collectors/collect_player_counts_pg.py`
- Deploy and test
- Run in parallel with old collector for 24 hours
- Switch over and archive DuckDB file

**Task 2.2: Migrate Item Metadata**
- Verify `items` table schema in PostgreSQL
- Update `collectors/update_items.py` to use PostgreSQL
- Deploy and test
- Archive DuckDB file

**Task 2.3: Create Shared Modules**
- Create `collectors/shared/__init__.py`
- Create `collectors/shared/metrics.py` (code in plan)
- Create `collectors/shared/config.py` (code in plan)
- Update collectors to use shared modules

### For Week 3 Tasks:

**Task 3.1: Deploy Prometheus**
- Create `collectors/prometheus/prometheus.yml` (config in plan)
- Update `collectors/docker-compose.yml` to add Prometheus service
- Deploy to server
- Verify scraping at http://150.136.170.128:9090/targets

**Task 3.2: Deploy Grafana**
- Update `collectors/docker-compose.yml` to add Grafana service
- Create provisioning configs for datasources and dashboards
- Deploy to server
- Create dashboards for collector metrics
- Access at http://150.136.170.128:3000

**Task 3.3: Set Up Alerting**
- Create `collectors/prometheus/alerts.yml` (rules in plan)
- Configure Alertmanager for Discord/Slack
- Test alerting by simulating failures

### For Week 4 Tasks:

**Task 4.1: Create Deployment Scripts**
- Create `deploy_collectors.sh` (script in plan)
- Create `scripts/check_collectors.sh` (script in plan)
- Test deployment script
- Document usage

**Task 4.2: Comprehensive Testing**
- Run full test checklist (in plan)
- Monitor all services for 24 hours
- Verify no data gaps
- Check all metrics

**Task 4.3: Update Documentation**
- Update `CLAUDE.md` with collector info
- Verify all READMEs are accurate
- Create operational runbook

## Success Criteria

1. ✅ Disk usage drops from 85% to <60%
2. ✅ All services using PostgreSQL (no DuckDB)
3. ✅ No hardcoded credentials in code
4. ✅ Prometheus + Grafana operational
5. ✅ Zero data collection gaps during migration
6. ✅ Single-command deployment working
7. ✅ Complete documentation

## Important Notes

- **All scripts are already written in the implementation plan** - you don't need to design them, just execute
- **The plan file has complete code examples** for all scripts and configurations
- **Safety first**: Always backup before database changes
- **Monitor actively**: Check logs and metrics after each deployment
- **Test thoroughly**: Use `--dry-run` modes when available
- **Rollback ready**: Have rollback commands prepared before each change

## Testing & Verification

After each task:
1. Check Docker container health: `docker ps`
2. Verify data freshness: Query latest timestamps
3. Check logs: `docker-compose logs -f [service]`
4. Monitor metrics: Check Prometheus endpoints
5. Verify no gaps: Query data for the past hour

## Rollback Procedures

**If compression fails**:
```sql
SELECT decompress_chunk(i) FROM show_chunks('price_data_5min') i;
```

**If Docker deployment fails**:
```bash
docker-compose down
git checkout docker-compose.yml
docker-compose up -d
```

**If collector migration fails**:
- Stop new collector
- Restart old DuckDB-based service
- Investigate logs
- Fix and retry

## Current Status

- ✅ All analysis complete
- ✅ All code written in plan
- ✅ All collectors copied to local repository
- ✅ Documentation created
- ⏳ Ready to begin Week 1 implementation

## Agent Instructions

You are implementing a well-planned infrastructure migration. **Follow the implementation plan exactly** - all scripts, SQL, and code are provided. Your job is to:

1. **Execute the tasks in order** following the 4-week timeline
2. **Use the exact scripts provided** in the plan file
3. **Verify each step** using the verification commands provided
4. **Monitor actively** and report any issues
5. **Update the todo list** as tasks are completed
6. **Document any deviations** from the plan

**Start with Week 1, Day 1**: Stop the redundant collector and fix the dashboard health check.

**Reference the plan file** at `docs/IMPLEMENTATION_PLAN.md` for all implementation details, scripts, and verification steps.

**Do not make changes without verification**. After each significant change, run the verification commands and monitor for at least 10 minutes before proceeding.

Good luck! The planning is done - now it's execution time.
