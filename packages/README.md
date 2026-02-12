# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Quick Start (Repo Root)

```bash
npm install

# Web app (Astro)
npm run dev:web

# Recommendation engine API
npm run dev:engine
```

## Common Workspace Scripts

```bash
npm run test       # run tests in all workspaces that define them
npm run lint       # run linters in all workspaces that define them
npm run typecheck  # run typechecks in all workspaces that define them
```
