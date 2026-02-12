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

Generate a local auth secret if you don't have one yet:

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
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

### Example `.env.local`

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/gept
BETTER_AUTH_SECRET=replace-with-generated-secret
BETTER_AUTH_URL=http://localhost:4321
PUBLIC_APP_URL=http://localhost:4321
SITE=http://localhost:4321
```

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
