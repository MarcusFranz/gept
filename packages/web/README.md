# GePT Web

Web frontend for GePT (Grand Exchange Prediction Tool). This Astro app serves the user-facing site, authentication, and trade alert flows.

## Quick Start

```bash
# From repo root
npm install

# Configure environment
cp packages/web/.env.example packages/web/.env.local

# Start the web app
npm run dev:web
```

The dev server runs on `http://localhost:4321` by default.

Requirements:
- Node.js 20+ (see `packages/web/package.json`)

## Environment

Environment variables live in `packages/web/.env.local`. Use `packages/web/.env.example` as the template.

Required configuration:
- `DATABASE_URL` for the Neon Postgres connection
- `BETTER_AUTH_SECRET`, `BETTER_AUTH_URL`, and `PUBLIC_APP_URL` for auth (set the URL values to the same origin as your dev server)
- `SITE` for canonical URLs and SEO metadata

Optional configuration:
- `RESEND_API_KEY` and `EMAIL_FROM` for password reset emails
- `PREDICTION_API` to use live predictions instead of mock data

Active-trade price alerts additionally require `ENGINE_WEBHOOK_URL` and `WEBHOOK_SECRET`.

## Scripts

From the repo root:

| Command | Action |
| --- | --- |
| `npm run dev:web` | Start Astro dev server |
| `npm run build:web` | Build production assets |

From `packages/web`:

| Command | Action |
| --- | --- |
| `npm run dev` | Start Astro dev server |
| `npm run build` | Build production assets |
| `npm run preview` | Preview production build |
| `npm run start` | Run the Node server entry |

## Project Structure

```text
packages/web/
├── migrations/
├── public/
├── src/
├── astro.config.mjs
└── package.json
```

- `src/` contains Astro pages and UI components.
- `migrations/` contains SQL migrations for the web database.
