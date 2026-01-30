# Security Audit Report — GePT Repository

**Date:** 2026-01-30
**Scope:** Full codebase security review (code, configuration, dependencies, infrastructure)

## Executive Summary

The codebase demonstrates solid security fundamentals — parameterized SQL queries, proper auth middleware, IDOR protections, and no committed secrets. However, several issues across 4 severity levels were identified that should be addressed.

---

## CRITICAL (Fix Immediately)

### 1. SSH Host Key Verification Disabled
- **File:** `deploy-ampere.sh:49`
- **Issue:** `StrictHostKeyChecking=no` enables man-in-the-middle attacks during deployment
- **Fix:** Use `StrictHostKeyChecking=accept-new` or pre-populate `known_hosts`

### 2. Database Password Exposed in CLI Arguments
- **File:** `deploy-ampere.sh:113`
- **Issue:** `PGPASSWORD=$DB_PASS psql ...` is visible in `ps aux` and logs
- **Fix:** Use a `.pgpass` file with chmod 600 permissions

### 3. Password Reset URL Logged with User Email
- **File:** `packages/web/src/lib/auth.ts:35,40-42`
- **Issue:** Reset tokens logged to console — can leak via log aggregation
- **Fix:** Remove all `console.log` of reset URLs; log only a redacted identifier

### 4. Password Reset Email Service Not Implemented
- **File:** `packages/web/src/lib/auth.ts:32-43`
- **Issue:** Reset tokens are only logged, never delivered — feature is non-functional
- **Fix:** Integrate with SendGrid, Resend, or AWS SES

### 5. Unaudited Sudo with Wildcard Copies
- **File:** `deploy-ampere.sh:224-236`
- **Issue:** `sudo cp ... *.service ... 2>/dev/null || true` copies any file matching the glob with error suppression
- **Fix:** Use explicit file paths; remove error suppression on critical operations

---

## HIGH (Fix This Sprint)

### 6. Rate Limiting Fails Open
- **File:** `packages/web/src/lib/rate-limit.ts:100-103,127-129,152-154`
- **Issue:** When Redis is unavailable, rate limiting is bypassed entirely
- **Fix:** Implement fail-closed approach or local memory fallback

### 7. Rate Limit Bypass via X-Forwarded-For Spoofing
- **File:** `packages/web/src/lib/rate-limit.ts:198-213`
- **Issue:** Signup rate limiting trusts the `X-Forwarded-For` header, which can be spoofed
- **Fix:** Only trust IP from the platform-specific header (Vercel sets this correctly when deployed)

### 8. Engine Rate Limit Key Spoofable
- **File:** `packages/engine/src/api.py:285-315`
- **Issue:** Rate limiting keys derived from untrusted `X-User-Hash` header
- **Fix:** Only accept user hash from authenticated session or signed tokens

### 9. API Key Comparison Not Timing-Safe
- **File:** `packages/engine/src/api.py:376`
- **Issue:** Uses `!=` for API key comparison instead of constant-time comparison
- **Fix:** Use `secrets.compare_digest(x_api_key, config.internal_api_key)`

### 10. Services Bound to 0.0.0.0
- **Files:** `packages/engine/Dockerfile:28`, `infra/systemd/gept-engine.service:13`, `packages/model/docker-compose.yml:62`
- **Issue:** API server and TensorBoard accessible from all interfaces on public cloud
- **Fix:** Bind to `127.0.0.1`; use a reverse proxy for external access

### 11. Missing Systemd Hardening
- **Files:** `infra/systemd/*.service`
- **Issue:** No `NoNewPrivileges`, `ProtectSystem`, `PrivateTmp`, etc.
- **Fix:** Add security directives per CIS benchmarks

### 12. Docker Containers Run as Root
- **Files:** `packages/engine/Dockerfile`, `packages/model/Dockerfile.*`
- **Issue:** 4 of 5 Dockerfiles lack a `USER` directive
- **Fix:** Add non-root user and `USER` directive to each Dockerfile

### 13. SSH Private Key Mounted into Docker Container
- **File:** `packages/model/docker-compose.yml:75`
- **Issue:** Private key mounted at `/root/.ssh/` with root access
- **Fix:** Use SSH agent socket or Docker secrets

### 14. Docker Images Use `latest` Tag
- **File:** `packages/model/collectors/docker-compose.yml:181,214,248`
- **Issue:** `prom/prometheus:latest`, `grafana/grafana:latest`, `prom/alertmanager:latest`
- **Fix:** Pin to specific version tags or digests

### 15. No Docker Resource Limits
- **File:** `packages/engine/docker-compose.yml`
- **Issue:** No memory/CPU limits configured
- **Fix:** Add `deploy.resources.limits` section

---

## MEDIUM

| # | Issue | Location |
|---|-------|----------|
| 16 | CORS `allow_headers=["*"]` with `allow_credentials=True` | `packages/engine/src/api.py:337` |
| 17 | Dashboard CORS wildcard (`Access-Control-Allow-Origin: *`) | `packages/model/collectors/dashboard.py:156` |
| 18 | Dashboard XSS via `innerHTML` with API response data | `packages/model/collectors/dashboard.py:232` |
| 19 | Dashboard SQL uses f-strings (safe today but fragile) | `packages/model/collectors/dashboard.py:60-75` |
| 20 | Session cookie security attrs not explicitly configured | `packages/web/src/lib/auth.ts:56-72` |
| 21 | Email enumeration possible via password reset timing | `packages/web/src/lib/rate-limit.ts:149-168` |
| 22 | Server IP `150.136.170.128` hardcoded throughout codebase | Multiple files |
| 23 | `psycopg2-binary` used instead of source `psycopg2` | 3 requirements.txt files |
| 24 | Python version mismatch (CI: 3.12, Docker: 3.10/3.11) | `.github/workflows/ci.yml` vs Dockerfiles |
| 25 | No `npm audit` or `pip audit` in CI pipeline | `.github/workflows/ci.yml` |
| 26 | No Dependabot or automated dependency scanning | Missing `.github/dependabot.yml` |
| 27 | `rsync --delete` in deployment — accidental data loss risk | `deploy-ampere.sh:150-169` |

---

## LOW

| # | Issue | Location |
|---|-------|----------|
| 28 | Test-only mock credentials in test files | `packages/engine/tests/` |
| 29 | Unpinned Python deps in model API requirements | `packages/model/api/requirements.txt` |
| 30 | Missing `.dockerignore` in collectors | `packages/model/collectors/` |

---

## Positive Findings (Secure Patterns)

- **No committed secrets** — `.gitignore` properly excludes `.env`, `.secrets/`, `.pem` files
- **Parameterized SQL everywhere** in the main app (SQLAlchemy, Neon template literals)
- **Subprocess calls use list args** — no `shell=True` injection risk
- **Webhook signature verification** uses `hmac.compare_digest()` (timing-safe)
- **IDOR protection** — all trade/user endpoints verify ownership
- **Security headers in middleware** — `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`
- **GitHub Actions uses secrets** for deployment credentials
- **`path-to-regexp` vulnerability mitigated** via npm overrides in `package.json`

---

## Recommended Priority Actions

1. **Immediate**: Remove password reset URL logging, fix SSH `StrictHostKeyChecking`, use `.pgpass` instead of CLI `PGPASSWORD`
2. **This sprint**: Change rate limiting to fail-closed, use `secrets.compare_digest()` for API keys, bind services to `127.0.0.1`, add `USER` to Dockerfiles, add systemd hardening
3. **Next sprint**: Implement email service for password reset, add `npm audit`/`pip audit` to CI, pin Docker images, add Dependabot, configure explicit cookie security attributes
4. **Backlog**: Fix dashboard XSS/CORS, remove hardcoded IPs, add Docker resource limits
