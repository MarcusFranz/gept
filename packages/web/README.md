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

Minimal `.env.local` example:

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/gept
BETTER_AUTH_SECRET=replace-with-random-token-at-least-32-chars
BETTER_AUTH_URL=http://localhost:4321
PUBLIC_APP_URL=http://localhost:4321
SITE=http://localhost:4321
```

## Environment Variables

Required:
- `DATABASE_URL` - Neon/Postgres connection string
- `BETTER_AUTH_SECRET` - auth signing secret (min 32 chars)
- `BETTER_AUTH_URL` - callback base URL (local: `http://localhost:3000`)
- `PUBLIC_APP_URL` - public app base URL
- `SITE` - canonical site URL for SEO

Optional:
- `RESEND_API_KEY` and `EMAIL_FROM` for transactional emails
- `PREDICTION_API` and `PREDICTION_API_KEY` for live recommendations
- `ENGINE_WEBHOOK_URL` and `WEBHOOK_SECRET` for active-trade alerts

### Local URLs

If you run everything on the Astro dev server, align these values to the dev port:

```bash
PUBLIC_APP_URL=http://localhost:4321
SITE=http://localhost:4321
BETTER_AUTH_URL=http://localhost:4321
```

If auth runs on a separate local server, keep `BETTER_AUTH_URL` pointed at that server instead.

### Troubleshooting

- **Auth redirect loop or callback mismatch**: make sure `BETTER_AUTH_URL`, `PUBLIC_APP_URL`, and `SITE` all point to the same local base URL (for example, `http://localhost:4321`), then restart the dev server.
- **Missing env errors**: confirm `packages/web/.env.local` exists and contains the required variables above.

## Scripts

```bash
npm run dev:web      # local dev server
npm run build:web    # production build
npm run preview --workspace=@gept/web
```

`npm run preview --workspace=@gept/web` requires a prior `npm run build:web`.

## Production Notes

- Uses server-side rendering via `@astrojs/node` and `@astrojs/vercel`.
- If `PREDICTION_API` is not set, the UI falls back to mock data.
