# Runbook: Deploy Recommendation Engine

## Purpose
Deploy a new version of the recommendation engine to production.

## Risk Level
MEDIUM - Service will briefly restart, but is stateless.

## Pre-requisites
- SSH access to Ampere server
- New code pushed to repository
- Tests passing in CI

## Pre-checks

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# 1. Check current version/status
curl http://localhost:8000/api/v1/health

# 2. Check disk space (need room for new deployment)
df -h /opt

# 3. Check no active issues
sudo journalctl -u recommendation-engine -n 20 --no-pager | grep -i error
```

## Procedure

### Step 1: Backup Current Version
```bash
cd /opt
sudo cp -r recommendation-engine recommendation-engine.bak
ls -la recommendation-engine.bak  # Verify backup exists
```

### Step 2: Pull New Code
```bash
cd /opt/recommendation-engine
sudo git fetch origin
sudo git status  # Check current branch
sudo git pull origin main
```

### Step 3: Update Dependencies (if needed)
```bash
# Check if requirements changed
git diff HEAD~1 requirements.txt

# If changed:
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### Step 4: Restart Service
```bash
sudo systemctl restart recommendation-engine
sleep 5
```

### Step 5: Verify Deployment
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Check logs for startup
sudo journalctl -u recommendation-engine -n 30 --no-pager

# Test a recommendation request (optional)
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"userId":"test","capital":50000000,"style":"hybrid","risk":"medium","slots":4}'
```

## Verification Checklist

- [ ] Health endpoint returns 200 OK
- [ ] Logs show "Application startup complete"
- [ ] No errors in recent logs
- [ ] Test request returns recommendations (if predictions available)

## Rollback

If deployment fails:

```bash
# 1. Restore backup
cd /opt
sudo rm -rf recommendation-engine
sudo mv recommendation-engine.bak recommendation-engine

# 2. Restart with old version
sudo systemctl restart recommendation-engine

# 3. Verify rollback
curl http://localhost:8000/api/v1/health
sudo journalctl -u recommendation-engine -n 20 --no-pager
```

## Post-deployment

1. Monitor logs for 5 minutes for any errors
2. Check Grafana dashboard for anomalies
3. Remove backup after 24 hours if stable:
   ```bash
   sudo rm -rf /opt/recommendation-engine.bak
   ```

## Troubleshooting

**Service won't start:**
```bash
# Check detailed error
sudo journalctl -u recommendation-engine -n 100 --no-pager

# Common issues:
# - Missing env vars: Check .env file
# - Port in use: sudo lsof -i :8000
# - Python errors: Check syntax/imports
```

**Health check fails:**
```bash
# Check if service is running
sudo systemctl status recommendation-engine

# Check if port is listening
sudo netstat -tlnp | grep 8000

# Check database connectivity
psql -h localhost -U osrs_user -d osrs_data -c "SELECT 1;"
```
