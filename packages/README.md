# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Monorepo Scripts (Run From Repo Root)

```bash
npm run dev:web        # web app dev server
npm run dev:engine     # recommendation engine API
npm run test           # run workspace tests (if present)
npm run lint           # lint all workspaces (if present)
npm run typecheck      # typecheck all workspaces (if present)
```
