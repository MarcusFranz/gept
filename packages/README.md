# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Local Entry Points

From the repo root you can install once and then launch services from the
workspace-level scripts:

```bash
npm install
npm run dev:web
npm run dev:engine
```

Each package has its own README with service-specific setup and environment
variables (some packages include `.env.example` templates).
