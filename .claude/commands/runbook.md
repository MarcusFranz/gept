# Run a Runbook

Execute a specific infrastructure runbook.

$file:/Users/marcusfranz/Documents/gept/agents/infra-agent.md

**Available runbooks:**

| Command | Risk | Description |
|---------|------|-------------|
| check-health | NONE | Full system health verification |
| check-inference | NONE | Verify ML inference pipeline |
| deploy-engine | MEDIUM | Deploy new engine version |
| deploy-model | MEDIUM | Deploy trained models |
| restart-engine | LOW | Restart recommendation engine |
| restart-collectors | LOW | Restart data collectors |
| restart-postgres | HIGH | Restart PostgreSQL (dangerous) |
| rollback-engine | LOW | Revert to previous engine version |

**Usage:** `/runbook <name>`

Which runbook should I execute? Provide the name (e.g., "check-health") or describe what you need.
