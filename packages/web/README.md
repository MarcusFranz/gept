# GePT Web App

Astro + SolidJS frontend for OSRS Grand Exchange flipping recommendations.

## Quick Start

```bash
# From repo root
npm install
npm run dev:web
```

The dev server runs at `http://localhost:4321`.

## Requirements

- Node.js 20+
- Environment variables in `packages/web/.env.local`

Create your local env file:

```bash
cp packages/web/.env.example packages/web/.env.local
```

If you run `npm run dev:web`, the app defaults to `http://localhost:4321`.
Set your auth URLs to match the dev server origin to avoid callback mismatches.

## Environment Variables

Required:
- `DATABASE_URL` - Neon/Postgres connection string
- `BETTER_AUTH_SECRET` - auth signing secret (min 32 chars)
- `BETTER_AUTH_URL` - callback base URL (local: `http://localhost:4321`)
- `PUBLIC_APP_URL` - public app base URL (local: `http://localhost:4321`)
- `SITE` - canonical site URL for SEO

Optional:
- `RESEND_API_KEY` and `EMAIL_FROM` for transactional emails
- `PREDICTION_API` and `PREDICTION_API_KEY` for live recommendations
- `ENGINE_WEBHOOK_URL` and `WEBHOOK_SECRET` for active-trade alerts

Example local overrides:

```bash
BETTER_AUTH_URL=http://localhost:4321
PUBLIC_APP_URL=http://localhost:4321
PREDICTION_API=http://localhost:8000
PREDICTION_API_KEY=replace-with-random-token
```

## Scripts

```bash
npm run dev:web      # local dev server
npm run build:web    # production build
npm run preview --workspace=@gept/web
```

## Production Notes

- Uses server-side rendering via `@astrojs/node` and `@astrojs/vercel`.
- If `PREDICTION_API` is not set, the UI falls back to mock data.
