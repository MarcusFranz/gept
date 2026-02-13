# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Local Dev Quick Start

From the repo root:

```bash
npm install

# Frontend
npm run dev:web

# Engine API
npm run dev:engine
```

Common workspace helpers:

```bash
npm run test
npm run lint
npm run typecheck
```
