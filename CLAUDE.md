# GePT System - Orchestrator Context

This document provides complete system context for AI agents working on GePT. Load this file to understand the full architecture and coordinate work across packages.

## System Overview

GePT (Grand Exchange Prediction Tool) is an ML-powered OSRS trading assistant with three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                │
│         OSRS GE API, Wiki, Historical APIs                      │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    packages/model                               │
│  Data ingestion → Feature computation → Training → Inference    │
│  Servers: Hydra (primary), Ampere (failover)                    │
│  Output: predictions table (every 5 min)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    packages/engine                              │
│  Reads predictions → Applies user constraints → Recommendations │
│  Server: Ampere (FastAPI on port 8000)                          │
│  Output: REST API for web/Discord                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    packages/web                                 │
│  Astro/SolidJS frontend → User dashboard                        │
│  Server: Vercel (edge deployment)                               │
│  Database: Neon (user data)                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Package Responsibilities

### packages/model
- **Data Ingestion**: 6 collectors (5min, 1min, hourly, news, player count, items)
- **Training**: CatBoost models on Hydra server (314 items, 108 targets each)
- **Inference**: Batch predictions every 5 minutes
- **Database**: PostgreSQL + TimescaleDB on Ampere (426M+ price rows)

### packages/engine
- **Recommendation Logic**: Transforms predictions into actionable trades
- **User Personalization**: Style (active/hybrid/passive), risk levels, capital
- **Portfolio Optimization**: Slot allocation, crowding prevention
- **API**: FastAPI serving web and Discord bot

### packages/web
- **Frontend**: Astro + SolidJS
- **Auth**: Better Auth with Neon PostgreSQL
- **Features**: Trade tracking, portfolio stats, real-time alerts (SSE)

### packages/shared
- **Types**: Recommendation, Trade, Webhook contracts
- **Config**: Rate limits, cache TTLs, API endpoints
- **Database**: Connection factory, schemas
- **Utilities**: Formatting, tax calculation

## Data Flow

```
1. Collectors write price data → PostgreSQL (5min intervals)
2. Inference reads prices → computes features → runs models
3. Inference writes predictions → predictions table
4. Engine reads predictions → applies filters → returns recommendations
5. Web calls engine API → displays to user
6. User completes trade → webhook to engine → feedback loop
```

## Critical Contracts

### Model → Engine (predictions table)
```sql
CREATE TABLE predictions (
    time TIMESTAMPTZ NOT NULL,
    item_id INTEGER NOT NULL,
    hour_offset INTEGER NOT NULL,        -- 1-48 hours
    offset_pct DECIMAL(5,4) NOT NULL,    -- 0.0125-0.025
    fill_probability DECIMAL(7,6),       -- [0, 1]
    expected_value DECIMAL(8,6),
    buy_price DECIMAL(12,2),
    sell_price DECIMAL(12,2),
    confidence TEXT,                      -- low/medium/high
    model_version TEXT,
    PRIMARY KEY (time, item_id, hour_offset, offset_pct)
);
```

### Engine → Web (API)
```
POST /api/v1/recommendations
  Request: { userId, capital, style, risk, slots, excludeIds }
  Response: Recommendation[]

GET /api/v1/items/search?q={query}
  Response: ItemSearchResult[]

POST /api/v1/trade-outcome
  Request: { userId, itemId, buyPrice, sellPrice, quantity, profit }
  Response: { success: boolean }
```

## Orchestrator Workflow

When you receive a task:

1. **Understand scope**: Does it touch model, engine, web, or multiple?
2. **Check contracts**: Will changes affect the interfaces above?
3. **Delegate appropriately**:
   - Model work → spawn agent with `agents/model-agent.md`
   - Engine work → spawn agent with `agents/engine-agent.md`
   - Web work → spawn agent with `agents/web-agent.md`
   - Server work → spawn agent with `agents/infra-agent.md`
4. **Coordinate**: If crossing boundaries, update contracts first
5. **Verify**: Check that changes don't break downstream consumers

## Agent Spawning

Use the Task tool to delegate:

```
Task(
    subagent_type="general-purpose",
    prompt="""
    Context: [Read agents/model-agent.md]

    Task: [specific task description]

    Constraints:
    - Do not modify predictions table schema without coordination
    - Update shared/types if changing API contracts
    """,
    description="Model: [short description]"
)
```

## Server Infrastructure

See `docs/servers.md` for complete infrastructure details.

**Quick Reference:**
- Ampere: `ubuntu@150.136.170.128` (engine, PostgreSQL, collectors)
- Hydra: Primary training and inference server
- Vercel: Web frontend deployment
- Neon: User database (web package)

## Development Rules

1. **Never push directly to main** - Always use PRs
2. **Update shared/ first** - When changing contracts
3. **Test across boundaries** - Engine changes need web testing
4. **Document breaking changes** - In CHANGELOG.md
5. **Server changes go through infra-agent** - Never ad-hoc SSH

## Common Tasks

### Deploy new model
1. Model agent: Train and validate on Hydra
2. Model agent: Update model registry
3. Infra agent: Deploy to production (Hydra primary, Ampere failover)
4. Engine agent: Verify predictions flowing

### Add new API endpoint
1. Engine agent: Implement endpoint
2. Shared: Add types if new contract
3. Web agent: Integrate in frontend
4. Test end-to-end

### Fix prediction bug
1. Identify which layer (model inference? engine filtering? web display?)
2. Delegate to appropriate agent
3. Verify fix doesn't break other layers

## File Locations

```
gept/
├── packages/
│   ├── shared/           # Shared types, config, utilities
│   ├── model/            # ML pipeline (from GePT Model)
│   ├── engine/           # Recommendations API (from GePT Recommendation Engine)
│   └── web/              # Frontend (from gept-gg)
├── docs/
│   ├── servers.md        # Infrastructure reference
│   ├── architecture.md   # Detailed system design
│   └── data-flow.md      # How data moves through system
├── infra/
│   └── runbooks/         # Safe procedures for server operations
├── agents/               # Subagent context files
│   ├── model-agent.md
│   ├── engine-agent.md
│   ├── web-agent.md
│   └── infra-agent.md
├── .secrets/             # SSH keys, credentials (gitignored)
└── CLAUDE.md             # This file (orchestrator context)
```
