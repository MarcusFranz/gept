# Claude Code Instructions for gept-gg

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.


## Project Overview
GePT is a Grand Exchange flipping assistant for Old School RuneScape. This repo contains the Astro/SolidJS frontend web application.

## Related Repositories

### gept-recommendation-engine
- **Repo**: `MarcusFranz/gept-recommendation-engine`
- **Purpose**: Prediction API that serves flip recommendations
- **URL**: `<prediction-api-url>` (redacted)
- **Use this repo for**: Any API changes, new endpoints, database schema changes, ML model updates

When you need changes to the prediction API (new endpoints, response format changes, etc.), create an issue on `gept-recommendation-engine` rather than trying to work around it in the frontend.

### gept-discord-bot
- **Repo**: `MarcusFranz/gept-discord-bot`
- **Purpose**: Discord bot interface for GePT

## Architecture

```
┌─────────────────┐     ┌──────────────────────────┐
│   gept-gg       │────▶│  gept-recommendation-    │
│   (this repo)   │     │  engine (API server)     │
│   Astro/Solid   │     │  FastAPI + PostgreSQL    │
│   Vercel        │     │  Ampere Server           │
└─────────────────┘     └──────────────────────────┘
        │
        ▼
┌─────────────────┐
│  Upstash Redis  │
│  (caching)      │
└─────────────────┘
```

## Key Environment Variables
- `PREDICTION_API` - URL to the recommendation engine API
- `PREDICTION_API_KEY` - API key for authentication
- `UPSTASH_REDIS_REST_URL` - Redis cache URL
- `UPSTASH_REDIS_REST_TOKEN` - Redis auth token

## Caching Strategy
- Item metadata: 24h TTL (names, icons - rarely change)
- Price history: 5min TTL (for sparklines)
- Recommendations: 2min TTL (personalized)

## Known Issues / TODOs
- Sparklines not working - waiting on price history endpoint (issue #187 on recommendation-engine)
- Consider implementing lean API responses to reduce payload size
