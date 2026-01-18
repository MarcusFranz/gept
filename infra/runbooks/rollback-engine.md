# Runbook: Rollback Recommendation Engine

## Purpose
Revert the recommendation engine to a previous working version.

## Risk Level
LOW - Restores known-good state.

## When to Use
- New deployment introduced bugs
- Health checks failing after deploy
- Unexpected behavior in recommendations
- Performance degradation

## Pre-requisites
- SSH access to Ampere server
- Backup exists at `/opt/recommendation-engine.bak`

## Pre-checks

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# 1. Confirm backup exists
ls -la /opt/recommendation-engine.bak
# If no backup, cannot rollback via this method

# 2. Check current state
curl http://localhost:8000/api/v1/health
sudo journalctl -u recommendation-engine -n 20 --no-pager
```

## Procedure

### Step 1: Stop Current Service
```bash
sudo systemctl stop recommendation-engine
```

### Step 2: Swap Directories
```bash
cd /opt

# Move broken version aside
sudo mv recommendation-engine recommendation-engine.broken

# Restore backup
sudo mv recommendation-engine.bak recommendation-engine

# Verify
ls -la recommendation-engine/
```

### Step 3: Restart Service
```bash
sudo systemctl start recommendation-engine
sleep 5
```

### Step 4: Verify Rollback
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Check logs
sudo journalctl -u recommendation-engine -n 30 --no-pager

# Test recommendation (optional)
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"userId":"test","capital":50000000,"style":"hybrid","risk":"medium","slots":4}'
```

## Verification Checklist

- [ ] Health endpoint returns 200 OK
- [ ] Logs show clean startup
- [ ] No errors in logs
- [ ] Original issue is resolved

## Post-rollback

1. **Keep broken version for debugging:**
   ```bash
   # Don't delete yet - may need to analyze
   ls -la /opt/recommendation-engine.broken/
   ```

2. **Investigate the issue:**
   - Check logs in broken version
   - Compare configs between versions
   - Review recent code changes

3. **Clean up after investigation (24-48 hours):**
   ```bash
   sudo rm -rf /opt/recommendation-engine.broken
   ```

4. **Create new backup of working version:**
   ```bash
   sudo cp -r /opt/recommendation-engine /opt/recommendation-engine.bak
   ```

## If No Backup Exists

**Option A: Git rollback**
```bash
cd /opt/recommendation-engine
sudo git log --oneline -10  # Find last good commit
sudo git checkout <commit-hash>
sudo systemctl restart recommendation-engine
```

**Option B: Redeploy from known tag**
```bash
cd /opt/recommendation-engine
sudo git fetch --tags
sudo git checkout v1.2.3  # Known good version
sudo systemctl restart recommendation-engine
```

## Troubleshooting

**Rollback version also fails:**
- Check if issue is external (database, network)
- Review PostgreSQL connectivity
- Check prediction freshness

**Service won't start after rollback:**
```bash
# Check for config drift
diff /opt/recommendation-engine/.env /opt/recommendation-engine.broken/.env

# May need to restore .env from backup
sudo cp /opt/recommendation-engine.broken/.env /opt/recommendation-engine/.env
sudo systemctl restart recommendation-engine
```
