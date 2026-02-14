# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Common Monorepo Commands

Run from repo root:

```bash
npm install
npm run dev:web
npm run dev:engine
```

Other helpful scripts:

```bash
npm run test
npm run lint
npm run typecheck
```
