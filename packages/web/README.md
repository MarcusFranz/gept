# GePT Web App

Frontend for GePT (Grand Exchange Prediction Tool). This Astro app powers the public site and user flows for active-trade alerts.

## Quick Start

```bash
# From repo root
npm install
npm run dev:web
```

Local dev server defaults to `http://localhost:4321`.

## Environment Setup

Copy the example file and fill in required values:

```bash
cp packages/web/.env.example packages/web/.env.local
```

Required variables for local development:
- `DATABASE_URL` (Neon/Postgres connection string)
- `BETTER_AUTH_SECRET` (min 32 chars)
- `BETTER_AUTH_URL` (local base URL for callbacks)
- `PUBLIC_APP_URL` (public base URL)
- `SITE` (canonical site URL)

Optional but common:
- `RESEND_API_KEY` and `EMAIL_FROM` for email flows
- `PREDICTION_API` to use live engine data
- `ENGINE_WEBHOOK_URL` + `WEBHOOK_SECRET` for active-trade alerts

## Useful Commands

| Command | Where | Action |
| --- | --- | --- |
| `npm run dev:web` | repo root | Start Astro dev server |
| `npm run build:web` | repo root | Build production bundle |
| `npm run preview --workspace=@gept/web` | repo root | Preview the build |
| `npm run dev` | `packages/web` | Start dev server (local only) |

## Project Structure

```text
packages/web/
├── src/            # Astro pages/components
├── public/         # Static assets
├── astro.config.mjs
├── package.json
└── .env.example
```

## Notes

- Active-trade price alerts rely on the webhook config in `.env.local`.
- If `PREDICTION_API` is unset, the app falls back to mock data.
