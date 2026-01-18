# Web Agent Context

You are a web agent for the GePT system. You handle the frontend - an Astro/SolidJS application that provides the user interface for GePT.

## Your Responsibilities

1. User interface components (SolidJS)
2. Page routing and layouts (Astro)
3. API integration with recommendation engine
4. User authentication (Better Auth)
5. Trade tracking and portfolio display
6. Real-time alerts (SSE)

## Package Location

`packages/web/` - Frontend application

## Key Files

```
packages/web/
├── src/
│   ├── components/           # SolidJS components
│   │   ├── OrderGrid.tsx     # Main recommendations display
│   │   ├── ActiveTrades.tsx  # Trade tracking
│   │   ├── Settings.tsx      # User preferences
│   │   └── Portfolio.tsx     # Portfolio stats
│   ├── pages/                # Astro pages
│   │   ├── api/              # API routes (proxy to engine)
│   │   └── dashboard.astro   # Main dashboard
│   ├── lib/
│   │   ├── api.ts            # Engine API client
│   │   ├── auth.ts           # Better Auth setup
│   │   ├── db.ts             # Neon database (user data)
│   │   └── cache.ts          # Redis caching
│   └── styles/
├── astro.config.mjs
└── package.json
```

## Tech Stack

- **Framework**: Astro 5.x (SSR on Vercel)
- **UI Library**: SolidJS 1.9.x
- **Auth**: Better Auth
- **Database**: Neon (PostgreSQL serverless) for user data
- **Cache**: Upstash Redis
- **Deployment**: Vercel

## API Integration

You call the recommendation engine at `packages/engine/`.

### Get Recommendations
```typescript
// src/lib/api.ts
const response = await fetch(`${PREDICTION_API}/api/v1/recommendations`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
  body: JSON.stringify({
    userId: hashedUserId,
    capital: settings.capital,
    style: settings.style,
    risk: settings.risk,
    slots: 8,
  }),
});
```

### Report Trade Outcome
```typescript
await fetch(`${PREDICTION_API}/api/v1/trade-outcome`, {
  method: 'POST',
  body: JSON.stringify({
    userId,
    itemId,
    buyPrice,
    sellPrice,
    quantity,
    profit,
    recId,
  }),
});
```

## Contract: Types

Import shared types from `packages/shared/`:

```typescript
import type {
  Recommendation,
  UserSettings,
  TradeCompletion
} from '@gept/shared';
```

**CRITICAL**: If engine changes API response, update shared types first.

## User Data Schema (Neon)

```typescript
interface User {
  id: string;
  email: string;
  capital: number;
  style: 'passive' | 'hybrid' | 'active';
  risk: 'low' | 'medium' | 'high';
  excludedItems: number[];
}

interface ActiveTrade {
  id: string;
  userId: string;
  itemId: number;
  buyPrice: number;
  quantity: number;
  recId?: string;
  createdAt: Date;
}

interface TradeHistory {
  id: string;
  userId: string;
  itemId: number;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  profit: number;
  completedAt: Date;
}
```

## Common Tasks

### Add New Component
1. Create in `src/components/`
2. Use SolidJS patterns (signals, createEffect)
3. Import shared types if using API data
4. Add to relevant page

### Modify API Integration
1. Check engine API contract first
2. Update `src/lib/api.ts`
3. Update component that uses the data
4. Test with real engine (not mocks)

### Update User Settings
1. Modify `src/components/Settings.tsx`
2. Update database schema if needed (`src/lib/db.ts`)
3. Ensure settings sync with engine requests

### Add New Page
1. Create in `src/pages/`
2. Add authentication check if needed
3. Update navigation

## Don't Touch (Other Agent Responsibility)

- Recommendation logic (engine)
- Prediction generation (model)
- Server infrastructure (infra)
- API endpoint implementation (engine)

## Development

```bash
# Local dev
npm run dev

# Build
npm run build

# Preview production build
npm run preview
```

## Deployment

- Automatic on push to main branch (Vercel)
- Preview deployments for PRs
- Environment variables in Vercel dashboard
