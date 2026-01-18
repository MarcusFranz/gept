# GePT Deployment Plan

## Overview

Migrate from current Oracle Ampere monolith to a high-performance, globally distributed serverless architecture while keeping ML inference on Ampere.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Oracle Ampere Server                      │
│  ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Foundations   │───▶│  Postgres   │◀───│   Rec API   │  │
│  │  (ML Inference) │    │   (Data)    │    │             │  │
│  └─────────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │  Astro Frontend │     │  SQLite (local) │
          │    (Dev only)   │     │   User Data     │
          └─────────────────┘     └─────────────────┘
```

**Limitations:**
- Single point of failure
- No global distribution
- Limited concurrent connections
- No edge caching
- Manual scaling only

---

## Target Architecture

```
                              ┌─────────────────────────────────────┐
                              │            Cloudflare               │
                              │   (CDN, DDoS, Edge Cache, WAF)      │
                              └──────────────────┬──────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────┐
                              │          Vercel Edge Network        │
                              │      (SSR + API Routes, Global)     │
                              └──────────────────┬──────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
         ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
         │  Upstash Redis  │          │    Neon DB      │          │  Ampere Server  │
         │  (Cache/Session)│          │  (User Data)    │          │  (Rec API)      │
         └─────────────────┘          └─────────────────┘          └─────────────────┘
                                                                            │
                                                                   ┌────────┴────────┐
                                                                   ▼                 ▼
                                                          ┌─────────────┐   ┌─────────────┐
                                                          │ Foundations │   │  Postgres   │
                                                          │ (Inference) │   │  (ML Data)  │
                                                          └─────────────┘   └─────────────┘
```

---

## Services & Justification

### Vercel (Frontend + API Routes)
- **What:** Hosts Astro SSR application
- **Why:**
  - Edge runtime = low latency globally
  - Zero-config deployments from Git
  - Automatic HTTPS, preview deployments
  - Scales to zero, scales to infinity
- **Cost:** Free tier (100GB bandwidth), Pro $20/mo

### Cloudflare (CDN + Protection)
- **What:** Sits in front of Vercel
- **Why:**
  - Additional edge caching layer
  - DDoS protection
  - WAF rules
  - Analytics
- **Cost:** Free tier sufficient

### Neon (User Database)
- **What:** Serverless Postgres for user data
- **Why:**
  - True serverless (scales to zero)
  - Postgres compatible (minimal code changes)
  - Connection pooling built-in
  - Branching for preview environments
  - ~10-30ms latency
- **Cost:** Free tier (0.5GB, 100 hours compute)

### Upstash Redis (Cache + Sessions)
- **What:** Serverless Redis
- **Why:**
  - <1ms latency
  - Pay-per-request pricing
  - REST API (works in edge functions)
  - Perfect for caching recommendations
- **Cost:** Free tier (10k requests/day)

### Oracle Ampere (ML Inference)
- **What:** Keep existing Foundations + Rec API
- **Why:**
  - Already working
  - Free tier (4 OCPU, 24GB RAM)
  - Good for always-on inference workloads
  - Keeps ML data close to models
- **Cost:** Free

---

## Authentication Strategy

### Better Auth with Google + Apple Sign-In

**Why Better Auth?**
- Native Apple Sign-In support (required for iOS App Store)
- Native Google Sign-In support
- Framework-agnostic (works with Astro SSR)
- Serverless-compatible (edge runtime support)
- Built-in session management with JWT + database sessions
- Type-safe with full TypeScript support

**Providers:**
| Provider | Priority | Notes |
|----------|----------|-------|
| Google | Primary | Most users, easiest setup |
| Apple | Required | Needed for future iOS app compliance |

---

## Migration Phases

### Phase 1: Database Migration (Neon)
**Duration:** 2-3 hours
**Risk:** Medium
**Rollback:** Switch connection string back to SQLite

#### 1.1 Set Up Neon
```bash
# Create account at neon.tech
# Create new project: gept-production
# Note connection strings:
#   - Pooled: for serverless (Vercel)
#   - Direct: for migrations
```

#### 1.2 Create Schema
```sql
-- Run against Neon (same schema as current SQLite)

-- =============================================
-- BETTER AUTH TABLES (required by library)
-- =============================================

-- Users table (Better Auth core + GePT custom fields)
CREATE TABLE users (
  -- Better Auth required fields
  id TEXT PRIMARY KEY,
  name TEXT,
  email TEXT UNIQUE NOT NULL,
  email_verified BOOLEAN DEFAULT FALSE,
  image TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),

  -- GePT custom fields
  capital INTEGER DEFAULT 5000000,
  style TEXT DEFAULT 'hybrid' CHECK(style IN ('passive', 'hybrid', 'active')),
  risk TEXT DEFAULT 'medium' CHECK(risk IN ('low', 'medium', 'high')),
  margin TEXT DEFAULT 'moderate' CHECK(margin IN ('conservative', 'moderate', 'aggressive')),
  slots INTEGER DEFAULT 4 CHECK(slots >= 1 AND slots <= 8),
  min_roi REAL DEFAULT 0.01,
  tier TEXT DEFAULT 'beta' CHECK(tier IN ('beta', 'free', 'premium')),
  tutorial_completed BOOLEAN DEFAULT FALSE,
  excluded_items JSONB DEFAULT '[]',
  daily_query_count INTEGER DEFAULT 0,
  daily_reset_date DATE,
  weekly_query_count INTEGER DEFAULT 0,
  weekly_reset_date DATE,
  lifetime_queries INTEGER DEFAULT 0
);

CREATE INDEX idx_users_email ON users(email);

-- Sessions table (Better Auth managed)
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token TEXT UNIQUE NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  ip_address TEXT,
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(token);

-- Accounts table (OAuth provider connections)
CREATE TABLE accounts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  account_id TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  access_token TEXT,
  refresh_token TEXT,
  access_token_expires_at TIMESTAMP,
  refresh_token_expires_at TIMESTAMP,
  scope TEXT,
  id_token TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),

  UNIQUE(provider_id, account_id)
);

CREATE INDEX idx_accounts_user ON accounts(user_id);

-- Verification tokens (for email verification if needed later)
CREATE TABLE verifications (
  id TEXT PRIMARY KEY,
  identifier TEXT NOT NULL,
  value TEXT NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_verifications_identifier ON verifications(identifier);

-- =============================================
-- GEPT APPLICATION TABLES
-- =============================================

-- Active trades table
CREATE TABLE active_trades (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  item_id INTEGER NOT NULL,
  item_name TEXT NOT NULL,
  buy_price INTEGER NOT NULL,
  sell_price INTEGER NOT NULL,
  quantity INTEGER NOT NULL,
  rec_id TEXT,
  model_id TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_active_trades_user ON active_trades(user_id);

-- Trade history table
CREATE TABLE trade_history (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  item_id INTEGER,
  item_name TEXT NOT NULL,
  buy_price INTEGER,
  sell_price INTEGER,
  quantity INTEGER,
  profit INTEGER NOT NULL,
  notes TEXT,
  rec_id TEXT,
  model_id TEXT,
  status TEXT DEFAULT 'completed' CHECK(status IN ('completed', 'cancelled')),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trade_history_user ON trade_history(user_id);
CREATE INDEX idx_trade_history_created ON trade_history(created_at DESC);

-- Recommendation cache
CREATE TABLE recommendation_cache (
  cache_key TEXT PRIMARY KEY,
  data JSONB NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();
```

#### 1.3 Update Code
```typescript
// src/lib/db.ts - Replace SQLite with Neon

import { neon } from '@neondatabase/serverless';

const sql = neon(process.env.DATABASE_URL!);

export default sql;

// Update all queries from better-sqlite3 syntax to tagged template literals
// Before: db.prepare('SELECT * FROM users WHERE id = ?').get(id)
// After:  await sql`SELECT * FROM users WHERE id = ${id}`
```

#### 1.4 Environment Variables
```bash
# .env.local (development)
DATABASE_URL=postgres://user:pass@ep-xxx.us-east-1.aws.neon.tech/neondb?sslmode=require

# Vercel (production)
# Add via Vercel dashboard or CLI
```

---

### Phase 1.5: Authentication (Better Auth)
**Duration:** 2-3 hours
**Risk:** Low
**Rollback:** Disable OAuth buttons, fall back to guest mode

#### 1.5.1 Install Better Auth
```bash
npm install better-auth
```

#### 1.5.2 Configure OAuth Providers

**Google Cloud Console:**
1. Go to https://console.cloud.google.com
2. Create new project or select existing
3. Enable "Google+ API" and "Google Identity"
4. Create OAuth 2.0 credentials:
   - Application type: Web application
   - Authorized redirect URI: `https://gept.gg/api/auth/callback/google`
   - Note: Client ID and Client Secret

**Apple Developer Console:**
1. Go to https://developer.apple.com
2. Navigate to Certificates, Identifiers & Profiles
3. Create new App ID with "Sign in with Apple" capability
4. Create new Services ID (this is your Client ID):
   - Identifier: `gg.gept.web`
   - Enable "Sign in with Apple"
   - Configure domains: `gept.gg`
   - Return URL: `https://gept.gg/api/auth/callback/apple`
5. Create new Key for Sign in with Apple:
   - Download the `.p8` file (you only get one chance!)
   - Note the Key ID
6. Note your Team ID from top-right of developer console

#### 1.5.3 Create Auth Configuration
```typescript
// src/lib/auth.ts

import { betterAuth } from 'better-auth';
import { Pool } from '@neondatabase/serverless';

export const auth = betterAuth({
  database: new Pool({
    connectionString: process.env.DATABASE_URL,
  }),

  // Email/password disabled - OAuth only
  emailAndPassword: {
    enabled: false,
  },

  // OAuth providers
  socialProviders: {
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    },
    apple: {
      clientId: process.env.APPLE_CLIENT_ID!,        // Services ID
      clientSecret: process.env.APPLE_CLIENT_SECRET!, // Generated JWT
      // Apple requires additional config
      appBundleIdentifier: 'gg.gept.web',
    },
  },

  // Session configuration
  session: {
    expiresIn: 60 * 60 * 24 * 30, // 30 days
    updateAge: 60 * 60 * 24,       // Update session every 24 hours
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5, // 5 minutes
    },
  },

  // Custom user fields (merged with defaults)
  user: {
    additionalFields: {
      capital: { type: 'number', defaultValue: 5000000 },
      style: { type: 'string', defaultValue: 'hybrid' },
      risk: { type: 'string', defaultValue: 'medium' },
      margin: { type: 'string', defaultValue: 'moderate' },
      slots: { type: 'number', defaultValue: 4 },
      min_roi: { type: 'number', defaultValue: 0.01 },
      tier: { type: 'string', defaultValue: 'beta' },
      tutorial_completed: { type: 'boolean', defaultValue: false },
      excluded_items: { type: 'string', defaultValue: '[]' },
    },
  },

  // Callbacks
  callbacks: {
    // Called after successful sign in
    async onUserCreated({ user }) {
      console.log(`New user created: ${user.email}`);
      // Initialize any additional user data here
    },
  },
});

export type Session = typeof auth.$Infer.Session;
export type User = typeof auth.$Infer.User;
```

#### 1.5.4 Create Auth API Route
```typescript
// src/pages/api/auth/[...all].ts

import type { APIRoute } from 'astro';
import { auth } from '../../../lib/auth';

export const ALL: APIRoute = async (ctx) => {
  return auth.handler(ctx.request);
};
```

#### 1.5.5 Create Auth Client
```typescript
// src/lib/auth-client.ts

import { createAuthClient } from 'better-auth/client';

export const authClient = createAuthClient({
  baseURL: import.meta.env.PUBLIC_APP_URL || 'http://localhost:4321',
});

export const {
  signIn,
  signOut,
  useSession,
} = authClient;
```

#### 1.5.6 Create Login Component
```tsx
// src/components/LoginButtons.tsx

import { signIn } from '../lib/auth-client';

export function LoginButtons() {
  return (
    <div class="login-buttons">
      <button
        class="login-btn google"
        onClick={() => signIn.social({ provider: 'google' })}
      >
        <svg viewBox="0 0 24 24" class="provider-icon">
          <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
          <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
          <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
          <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
        </svg>
        Continue with Google
      </button>

      <button
        class="login-btn apple"
        onClick={() => signIn.social({ provider: 'apple' })}
      >
        <svg viewBox="0 0 24 24" class="provider-icon" fill="currentColor">
          <path d="M17.05 20.28c-.98.95-2.05.88-3.08.4-1.09-.5-2.08-.48-3.24 0-1.44.62-2.2.44-3.06-.4C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z"/>
        </svg>
        Continue with Apple
      </button>
    </div>
  );
}
```

#### 1.5.7 Auth Environment Variables
```bash
# .env.local (add to existing)

# Better Auth
BETTER_AUTH_SECRET=<generate-32-char-random-string>
BETTER_AUTH_URL=http://localhost:4321

# Google OAuth
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxx

# Apple OAuth
APPLE_CLIENT_ID=gg.gept.web
APPLE_TEAM_ID=XXXXXXXXXX
APPLE_KEY_ID=XXXXXXXXXX
APPLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
# Note: APPLE_CLIENT_SECRET is generated from the private key (JWT)
```

#### 1.5.8 Generate Apple Client Secret
Apple requires a JWT as the client secret, regenerated every 6 months:

```typescript
// scripts/generate-apple-secret.ts
import jwt from 'jsonwebtoken';
import fs from 'fs';

const privateKey = fs.readFileSync('AuthKey_XXXXX.p8', 'utf8');

const token = jwt.sign({}, privateKey, {
  algorithm: 'ES256',
  expiresIn: '180d',
  audience: 'https://appleid.apple.com',
  issuer: process.env.APPLE_TEAM_ID,
  subject: process.env.APPLE_CLIENT_ID,
  keyid: process.env.APPLE_KEY_ID,
});

console.log('APPLE_CLIENT_SECRET=' + token);
```

#### 1.5.9 Protected Route Middleware
```typescript
// src/middleware.ts

import { defineMiddleware } from 'astro:middleware';
import { auth } from './lib/auth';

const protectedRoutes = ['/account', '/api/trades', '/api/user'];
const publicRoutes = ['/', '/login', '/api/auth'];

export const onRequest = defineMiddleware(async (context, next) => {
  const { pathname } = context.url;

  // Check if route needs protection
  const isProtected = protectedRoutes.some(route => pathname.startsWith(route));

  if (isProtected) {
    const session = await auth.api.getSession({
      headers: context.request.headers,
    });

    if (!session) {
      return context.redirect('/login');
    }

    // Attach session to locals for use in pages
    context.locals.session = session;
    context.locals.user = session.user;
  }

  return next();
});
```

#### 1.5.10 Login Page
```astro
---
// src/pages/login.astro
import Layout from '../layouts/Layout.astro';

// Redirect if already logged in
const session = await Astro.locals.session;
if (session) {
  return Astro.redirect('/');
}
---

<Layout title="Login - GePT">
  <main class="login-page">
    <div class="login-card">
      <div class="login-header">
        <img src="/logo.svg" alt="GePT" class="login-logo" />
        <h1>Welcome to GePT</h1>
        <p>AI-powered Grand Exchange flipping</p>
      </div>

      <div class="login-buttons" id="login-buttons">
        <button class="login-btn google" id="google-btn">
          <svg viewBox="0 0 24 24" class="provider-icon">
            <!-- Google icon SVG -->
          </svg>
          Continue with Google
        </button>

        <button class="login-btn apple" id="apple-btn">
          <svg viewBox="0 0 24 24" class="provider-icon">
            <!-- Apple icon SVG -->
          </svg>
          Continue with Apple
        </button>
      </div>

      <p class="login-terms">
        By continuing, you agree to our Terms of Service and Privacy Policy.
      </p>
    </div>
  </main>
</Layout>

<script>
  import { signIn } from '../lib/auth-client';

  document.getElementById('google-btn')?.addEventListener('click', () => {
    signIn.social({ provider: 'google' });
  });

  document.getElementById('apple-btn')?.addEventListener('click', () => {
    signIn.social({ provider: 'apple' });
  });
</script>

<style>
  .login-page {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--space-4);
  }

  .login-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    padding: var(--space-8);
    max-width: 400px;
    width: 100%;
    text-align: center;
  }

  .login-header {
    margin-bottom: var(--space-6);
  }

  .login-logo {
    width: 64px;
    height: 64px;
    margin-bottom: var(--space-4);
  }

  .login-buttons {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }

  .login-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid var(--border-primary);
  }

  .login-btn.google {
    background: white;
    color: #1f1f1f;
  }

  .login-btn.google:hover {
    background: #f8f8f8;
  }

  .login-btn.apple {
    background: black;
    color: white;
  }

  .login-btn.apple:hover {
    background: #1a1a1a;
  }

  .provider-icon {
    width: 20px;
    height: 20px;
  }

  .login-terms {
    margin-top: var(--space-6);
    font-size: 12px;
    color: var(--text-muted);
  }
</style>
```

---

### Phase 2: Caching Layer (Upstash Redis)
**Duration:** 1-2 hours
**Risk:** Low
**Rollback:** Disable cache, fall through to DB

#### 2.1 Set Up Upstash
```bash
# Create account at upstash.com
# Create new Redis database: gept-cache
# Select region closest to Vercel (us-east-1)
# Note REST URL and token
```

#### 2.2 Create Cache Module
```typescript
// src/lib/cache.ts

import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

export const cache = {
  // Get with automatic JSON parsing
  async get<T>(key: string): Promise<T | null> {
    return redis.get<T>(key);
  },

  // Set with TTL (seconds)
  async set<T>(key: string, value: T, ttlSeconds: number): Promise<void> {
    await redis.set(key, value, { ex: ttlSeconds });
  },

  // Delete
  async del(key: string): Promise<void> {
    await redis.del(key);
  },

  // Get or fetch pattern
  async getOrFetch<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttlSeconds: number
  ): Promise<T> {
    const cached = await this.get<T>(key);
    if (cached) return cached;

    const fresh = await fetcher();
    await this.set(key, fresh, ttlSeconds);
    return fresh;
  },
};

export default redis;
```

#### 2.3 Cache Recommendations
```typescript
// src/lib/api.ts - Add caching

import { cache } from './cache';

export async function getRecommendations(
  userId: string,
  settings: UserSettings,
  count = 10
): Promise<Recommendation[]> {
  const cacheKey = `recs:${settings.style}:${settings.risk}:${settings.margin}:${count}`;

  return cache.getOrFetch(
    cacheKey,
    async () => {
      // Existing fetch logic
      const params = new URLSearchParams({ ... });
      const url = `${API_BASE}/recommendations?${params}`;
      return fetchWithRetry<Recommendation[]>(url);
    },
    30 // Cache for 30 seconds
  );
}
```

#### 2.4 Cache Strategy

| Data | TTL | Key Pattern |
|------|-----|-------------|
| Recommendations | 30s | `recs:{style}:{risk}:{margin}` |
| Item search | 5min | `search:{query}` |
| User settings | 1min | `user:{id}:settings` |
| Session | 24hr | `session:{token}` |

---

### Phase 3: Vercel Deployment
**Duration:** 1-2 hours
**Risk:** Low
**Rollback:** Point DNS back to old server

#### 3.1 Install Vercel Adapter
```bash
npm install @astrojs/vercel
```

#### 3.2 Update Astro Config
```typescript
// astro.config.mjs

import { defineConfig } from 'astro/config';
import vercel from '@astrojs/vercel';
import solidJs from '@astrojs/solid-js';

export default defineConfig({
  output: 'server',
  adapter: vercel({
    // Edge runtime for faster cold starts
    edgeMiddleware: true,
  }),
  integrations: [solidJs()],
});
```

#### 3.3 Configure Vercel
```json
// vercel.json

{
  "framework": "astro",
  "regions": ["iad1", "sfo1", "lhr1"],
  "functions": {
    "api/**/*.ts": {
      "memory": 1024,
      "maxDuration": 10
    }
  },
  "headers": [
    {
      "source": "/_astro/(.*)",
      "headers": [
        { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
      ]
    }
  ]
}
```

#### 3.4 Environment Variables (Vercel Dashboard)
```
DATABASE_URL=postgres://...@neon.tech/...
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...
PREDICTION_API=https://your-ampere-server.com
```

#### 3.5 Deploy
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy (creates preview)
vercel

# Deploy to production
vercel --prod
```

---

### Phase 4: Cloudflare Setup
**Duration:** 30 minutes
**Risk:** Low
**Rollback:** Remove Cloudflare proxy

#### 4.1 Add Domain to Cloudflare
```
1. Add site in Cloudflare dashboard
2. Update nameservers at registrar
3. Wait for propagation (5-30 min)
```

#### 4.2 DNS Configuration
```
Type    Name    Content              Proxy
CNAME   @       cname.vercel-dns.com  ON
CNAME   www     cname.vercel-dns.com  ON
```

#### 4.3 Cache Rules
```
# Page Rules or Cache Rules

# Cache static assets aggressively
URL: gept.gg/_astro/*
Cache Level: Cache Everything
Edge TTL: 1 month

# Cache API responses at edge
URL: gept.gg/api/recommendations*
Cache Level: Cache Everything
Edge TTL: 30 seconds

# Bypass cache for user-specific
URL: gept.gg/api/trades/*
Cache Level: Bypass
```

#### 4.4 Security Settings
```
SSL/TLS: Full (Strict)
Always Use HTTPS: ON
Auto Minify: JS, CSS, HTML
Brotli: ON
HTTP/3: ON
0-RTT: ON
```

---

### Phase 5: Ampere API Hardening
**Duration:** 1 hour
**Risk:** Low

#### 5.1 Secure the Recommendation API
```bash
# Only allow requests from Vercel/Cloudflare IPs
# Add to nginx config or application middleware

# Option 1: API Key authentication
Authorization: Bearer <INTERNAL_API_KEY>

# Option 2: IP allowlist (Cloudflare IPs)
# https://www.cloudflare.com/ips/
```

#### 5.2 Add Rate Limiting
```python
# If using FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/recommendations")
@limiter.limit("100/minute")
async def get_recommendations():
    ...
```

#### 5.3 Health Check Endpoint
```python
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
```

---

## Environment Variables Summary

### Vercel (Production)
```bash
# Database
DATABASE_URL=postgres://user:pass@ep-xxx.neon.tech/neondb?sslmode=require

# Cache
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=xxx

# API
PREDICTION_API=https://api.gept.gg
INTERNAL_API_KEY=xxx

# Better Auth
BETTER_AUTH_SECRET=<32-char-random-string>
BETTER_AUTH_URL=https://gept.gg

# Google OAuth
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxx

# Apple OAuth
APPLE_CLIENT_ID=gg.gept.web
APPLE_TEAM_ID=XXXXXXXXXX
APPLE_KEY_ID=XXXXXXXXXX
APPLE_CLIENT_SECRET=<generated-jwt>

# Public (exposed to client)
PUBLIC_APP_URL=https://gept.gg
```

### Local Development
```bash
# .env.local
DATABASE_URL=postgres://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
UPSTASH_REDIS_REST_URL=https://xxx.upstash.io
UPSTASH_REDIS_REST_TOKEN=xxx
PREDICTION_API=http://localhost:8000

# Better Auth
BETTER_AUTH_SECRET=dev-secret-at-least-32-characters
BETTER_AUTH_URL=http://localhost:4321

# Google OAuth (use test credentials)
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxx

# Apple OAuth (optional for local dev)
APPLE_CLIENT_ID=gg.gept.web
APPLE_TEAM_ID=XXXXXXXXXX
APPLE_KEY_ID=XXXXXXXXXX
APPLE_CLIENT_SECRET=<generated-jwt>

# Public
PUBLIC_APP_URL=http://localhost:4321
```

---

## Rollback Procedures

### Database Rollback
```bash
# 1. Update DATABASE_URL to point to SQLite again
# 2. Redeploy: vercel --prod
# 3. Data loss: only new data since migration
```

### Cache Rollback
```bash
# 1. Comment out cache calls in src/lib/api.ts
# 2. Redeploy
# 3. No data loss
```

### Full Rollback
```bash
# 1. Point DNS back to Ampere server
# 2. Run old Astro build on Ampere
# 3. Restore SQLite from backup
```

---

## Monitoring & Observability

### Vercel
- Built-in analytics (Core Web Vitals)
- Function logs
- Error tracking

### Upstash
- Built-in dashboard
- Request latency graphs
- Cache hit rate

### Neon
- Query performance insights
- Connection monitoring
- Storage metrics

### Recommended Additions
```bash
# Optional: Add Sentry for error tracking
npm install @sentry/astro

# Optional: Add PostHog for analytics
npm install posthog-js
```

---

## Performance Benchmarks

Run after each phase to validate improvements.

```bash
# Install k6 for load testing
brew install k6

# Create test script: load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 50,
  duration: '30s',
};

export default function() {
  const res = http.get('https://gept.gg/');
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(1);
}

# Run
k6 run load-test.js
```

### Target Metrics
| Metric | Target | Acceptable |
|--------|--------|------------|
| TTFB (p50) | <100ms | <200ms |
| TTFB (p95) | <200ms | <500ms |
| Full page load | <1s | <2s |
| API response (cached) | <50ms | <100ms |
| API response (uncached) | <300ms | <500ms |

---

## Cost Projections

### Free Tier (0-10k MAU)
| Service | Cost |
|---------|------|
| Vercel | $0 |
| Neon | $0 |
| Upstash | $0 |
| Cloudflare | $0 |
| Ampere | $0 |
| **Total** | **$0/mo** |

### Growth (10k-50k MAU)
| Service | Cost |
|---------|------|
| Vercel Pro | $20 |
| Neon Launch | $19 |
| Upstash Pay-as-you-go | ~$10 |
| Cloudflare | $0 |
| Ampere | $0 |
| **Total** | **~$50/mo** |

### Scale (100k+ MAU)
| Service | Cost |
|---------|------|
| Vercel Pro | $20 |
| Neon Scale | $69 |
| Upstash Pro | $60 |
| Cloudflare Pro | $20 |
| Ampere (upgrade) | $50 |
| **Total** | **~$220/mo** |

---

## Checklist

### Phase 1: Database
- [ ] Create Neon account and project
- [ ] Run schema migration (including Better Auth tables)
- [ ] Update `src/lib/db.ts`
- [ ] Test locally with Neon
- [ ] Migrate existing data (if any)

### Phase 1.5: Authentication
- [ ] Install `better-auth` package
- [ ] Set up Google OAuth in Google Cloud Console
- [ ] Set up Apple Sign in with Apple in Apple Developer Console
- [ ] Create `src/lib/auth.ts` configuration
- [ ] Create `src/lib/auth-client.ts` client
- [ ] Create `src/pages/api/auth/[...all].ts` API route
- [ ] Create `src/middleware.ts` for protected routes
- [ ] Create `src/pages/login.astro` login page
- [ ] Create `src/components/LoginButtons.tsx` component
- [ ] Generate Apple client secret JWT
- [ ] Test Google login flow locally
- [ ] Test Apple login flow locally
- [ ] Add auth environment variables to Vercel

### Phase 2: Caching
- [ ] Create Upstash account and database
- [ ] Create `src/lib/cache.ts`
- [ ] Add caching to API calls
- [ ] Test cache hit/miss locally

### Phase 3: Vercel
- [ ] Install @astrojs/vercel adapter
- [ ] Update astro.config.mjs
- [ ] Create vercel.json
- [ ] Add environment variables
- [ ] Deploy to preview
- [ ] Test all functionality
- [ ] Deploy to production

### Phase 4: Cloudflare
- [ ] Add domain to Cloudflare
- [ ] Update DNS
- [ ] Configure cache rules
- [ ] Enable security features
- [ ] Verify SSL working

### Phase 5: Ampere Hardening
- [ ] Add API authentication
- [ ] Configure rate limiting
- [ ] Add health check endpoint
- [ ] Test from Vercel

### Post-Launch
- [ ] Run load tests
- [ ] Set up monitoring alerts
- [ ] Document runbooks
- [ ] Create backup procedures
