# Runbook: Restart Recommendation Engine

## Purpose
Safely restart the recommendation engine service.

## Risk Level
LOW - Engine is stateless, auto-reconnects to database.

## Pre-checks

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# 1. Check current status
sudo systemctl status recommendation-engine

# 2. Note current uptime (for comparison after restart)
# 3. Verify health before restart
curl http://localhost:8000/api/v1/health
```

## Procedure

```bash
# Restart service
sudo systemctl restart recommendation-engine

# Wait for startup
sleep 5

# Verify health
curl http://localhost:8000/api/v1/health

# Check logs for errors
sudo journalctl -u recommendation-engine -n 20 --no-pager
```

## Verification

- [ ] Health endpoint returns 200 OK
- [ ] Logs show "Application startup complete"
- [ ] No errors in recent logs
- [ ] Uptime reset (confirms restart happened)

## Rollback

If health check fails after restart:

```bash
# Check detailed logs
sudo journalctl -u recommendation-engine -n 50 --no-pager

# If config issue, restore previous config
cp /opt/recommendation-engine.bak/.env /opt/recommendation-engine/.env
sudo systemctl restart recommendation-engine

# If code issue, rollback deployment
cd /opt/recommendation-engine
./rollback.sh  # If available
```

## Post-action

Document in your response:
- Time of restart
- Reason for restart
- Status after restart
- Any anomalies observed
