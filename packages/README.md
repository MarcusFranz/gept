# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Workspace Commands

From the repo root, use the workspace scripts:

```bash
npm run dev:web
npm run dev:engine
npm run lint
npm run typecheck
npm run test
```

Run a script in a specific package workspace:

```bash
npm run dev --workspace=@gept/web
```
