---
name: health
description: Use when checking system health - runs infrastructure, inference, application, and data quality checks
---

# System Health Check Workflow

You are the GePT orchestrator running a health check.

## Your Role
You verify all systems are operational and report status.

## Context
$file:/Users/marcusfranz/Documents/gept/CLAUDE.md
$file:/Users/marcusfranz/Documents/gept/docs/servers.md

## Workflow Pipeline

### Phase 1: Infrastructure Check (spawn Infra agent)
Spawn Task agent to execute check-health runbook:
- Server connectivity (Ampere, Hydra)
- Service status (PostgreSQL, Engine, Collectors)
- Disk space and resources
- Process health

### Phase 2: Inference Pipeline Check
Spawn Task agent to execute check-inference runbook:
- Model availability
- Prediction freshness
- Inference latency
- Fallback status

### Phase 3: Application Check
Verify application layer:
- Engine API responding
- Web frontend accessible
- Database connections working
- Webhook delivery

### Phase 4: Data Quality Check
Verify data integrity:
- Recent predictions exist
- Collector data flowing
- No stale data alerts

### Phase 5: Report

Present health dashboard:

```
┌─────────────────────────────────────────┐
│          GePT Health Status             │
├─────────────────────────────────────────┤
│ Infrastructure                          │
│   Ampere:     [OK/WARN/FAIL]           │
│   Hydra:      [OK/WARN/FAIL]           │
│   PostgreSQL: [OK/WARN/FAIL]           │
├─────────────────────────────────────────┤
│ Services                                │
│   Engine:     [OK/WARN/FAIL]           │
│   Collectors: [OK/WARN/FAIL]           │
│   Inference:  [OK/WARN/FAIL]           │
├─────────────────────────────────────────┤
│ Data                                    │
│   Predictions: [Fresh/Stale]           │
│   Last Update: [timestamp]             │
└─────────────────────────────────────────┘
```

## Alert Thresholds

**FAIL**: Service down, no response, critical error
**WARN**: Degraded performance, stale data, high resource usage
**OK**: Normal operation

## Begin

Running full system health check...
