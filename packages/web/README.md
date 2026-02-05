# GePT Web App

Frontend for GePT (Grand Exchange Prediction Tool). Serves the website and API routes for account/auth and trade workflows.

## Quick Start

```bash
# From repo root
npm install
npm run dev:web
```

Or run directly from this package:

```bash
cd packages/web
npm install
npm run dev
```

Astro will start the dev server (default `http://localhost:4321`).

## Environment

Copy the template and set required values:

```bash
cp .env.example .env.local
```

Key variables to double-check for local dev:

- `PUBLIC_APP_URL` and `BETTER_AUTH_URL` should match your dev server URL
- `DATABASE_URL` must point at your Postgres instance
- `PREDICTION_API` should point at the engine API (uses mock data if unset)
- `ENGINE_WEBHOOK_URL` and `WEBHOOK_SECRET` are required for active-trade alerts

## Common Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start Astro dev server |
| `npm run build` | Build the production bundle |
| `npm run preview` | Preview the production build |
| `npm run start` | Run the built server (`dist/server/entry.mjs`) |

## Related Docs

- Engine API reference: `packages/engine/docs/API.md`
- Engine webhook details: `packages/engine/docs/DISCORD_BOT_INTEGRATION.md`
