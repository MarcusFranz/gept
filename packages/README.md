# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Common Commands

Run these from the repo root:

```bash
npm run dev:web        # Web app dev server
npm run dev:engine     # Engine API (FastAPI)
npm run test           # Run all workspace tests
npm run lint           # Lint all workspaces
npm run typecheck      # Type-check all workspaces
```
