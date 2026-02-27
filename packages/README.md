# Packages

This repo is organized by deployment unit:

- `packages/web`: Vercel-deployed Astro site
- `packages/engine`: Recommendation Engine API (Docker Compose on server)
- `packages/collectors`: Data collection + monitoring stack (Docker Compose on server)
- `packages/inference`: Batch inference job (Docker one-shot on a systemd timer)
- `packages/shared`: Shared TypeScript utilities/types used by web

Server `systemd` units live in `infra/systemd/`.

## Shared Package

`packages/shared` exports common types and helpers used by `packages/web`. Import them via the workspace name in the web app (see `packages/shared/package.json`).

## Workspace Commands

From the repo root, use the workspace scripts defined in `package.json`:

- `npm run dev:web` - run the Astro dev server
- `npm run dev:engine` - run the FastAPI engine via `uvicorn`
- `npm run test` - run tests in all workspaces that define them
- `npm run lint` - lint all workspaces that define it
- `npm run typecheck` - typecheck all workspaces that define it
