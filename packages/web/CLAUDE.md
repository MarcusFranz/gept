# Claude Code Instructions for gept-gg

## Project Overview
GePT is a Grand Exchange flipping assistant for Old School RuneScape. This repo contains the Astro/SolidJS frontend web application.

## Related Repositories

### gept-recommendation-engine
- **Repo**: `MarcusFranz/gept-recommendation-engine`
- **Purpose**: Prediction API that serves flip recommendations
- **URL**: `http://150.136.170.128:8000` (Ampere server)
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
