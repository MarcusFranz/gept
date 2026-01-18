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

## Workflows (User-Facing Commands)

Users invoke high-level workflows via slash commands:

| Command | Purpose |
|---------|---------|
| `/feature` | Design and implement a new feature |
| `/bugfix` | Diagnose and fix a bug |
| `/refactor` | Safe refactoring with tests |
| `/deploy` | Deploy to production using runbooks |
| `/health` | System health check |
| `/incident` | Production incident response |

Each workflow defines a pipeline of agents that process the work.

## Agent Pipeline Architecture

```
User Request
     │
     ▼
┌─────────────────────┐
│   ORCHESTRATOR      │  ← You (full system context)
│   Understands scope │
│   Coordinates work  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   PLANNER AGENT     │  ← Expands request to design doc
│   Architecture      │     with requirements, steps, contracts
│   Task breakdown    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   DISPATCHER        │  ← Analyzes dependencies
│   (Orchestrator)    │     Determines parallelization
└─────────┬───────────┘
          │
    ┌─────┴─────┬─────────┬─────────┐
    ▼           ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ Model │ │Engine │ │ Web   │ │ Infra │  ← Specialized workers
│Worker │ │Worker │ │Worker │ │Worker │     (parallel when possible)
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │
    └─────────┴────┬────┴─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   QA AGENT      │  ← Tests, integration, validation
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │   ORCHESTRATOR  │  ← Review results, report to user
          └─────────────────┘
```

## Agent Spawning Patterns

### Planning Phase
```
Task(
    subagent_type="Plan",
    prompt="Create a design document for: [feature description]
           Consider: affected packages, contracts, risks
           Output: markdown design doc",
    description="Plan: feature design"
)
```

### Worker Phase (spawn in parallel when no dependencies)
```
Task(
    subagent_type="general-purpose",
    prompt="""
    You are a Model Worker for GePT.

    Context: [include agents/model-agent.md content]
    Design: [include design doc content]

    Your tasks:
    1. [specific task from design]
    2. [specific task from design]

    Constraints:
    - Do not modify predictions table schema
    - Update shared/types if changing contracts
    """,
    description="Model: implement feature X"
)
```

### QA Phase
```
Task(
    subagent_type="general-purpose",
    prompt="Run tests and verify:
           1. All test suites pass
           2. Acceptance criteria met: [list from design]
           3. No regressions introduced
           4. Contracts maintained",
    description="QA: verify implementation"
)
```

## Subagent Context Files

Located in `agents/` - include content in worker prompts:

| File | Use For |
|------|---------|
| `model-agent.md` | ML pipeline, training, inference, collectors |
| `engine-agent.md` | API endpoints, recommendations, business logic |
| `web-agent.md` | Frontend, UI components, auth |
| `infra-agent.md` | Server operations (ALWAYS use runbooks) |

## Parallelization Rules

**Can run in parallel:**
- Independent package changes (model + web if no shared contract changes)
- Multiple bug fixes in different files
- Read-only exploration tasks

**Must run sequentially:**
- Contract changes → then consumers
- Database schema → then code using it
- Auth changes → then features depending on auth

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
