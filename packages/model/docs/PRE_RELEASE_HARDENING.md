# GePT Pre-Release Hardening Plan

**Project URL:** https://github.com/users/MarcusFranz/projects/6

This document outlines the implementation plan for 31 issues identified during the pre-release codebase review. Issues are organized into phases with clear dependencies and parallelization opportunities.

---

## Executive Summary

| Phase | Issues | Effort | Parallelizable |
|-------|--------|--------|----------------|
| Phase 0: Critical Blockers | 3 | ~4 hrs | Yes (2 tracks) |
| Phase 1: Security & Stability | 3 | ~6 hrs | Yes (2 tracks) |
| Phase 2: Error Handling & Resilience | 5 | ~10 hrs | Yes (2 tracks) |
| Phase 3: Configuration & Monitoring | 8 | ~12 hrs | Yes (4 tracks) |
| Phase 4: Data Quality | 4 | ~8 hrs | Yes (2 tracks) |
| Phase 5: Operations & DevOps | 4 | ~10 hrs | Partially |
| Phase 6: Code Quality & Performance | 4 | ~8 hrs | Yes (3 tracks) |
| **Total** | **31** | **~58 hrs** | |

---

## Phase 0: Critical Blockers (MUST FIX BEFORE DEPLOY)

These issues represent security vulnerabilities or data loss risks that must be fixed immediately.

### Issues

| Issue | Title | Effort | Parallel Group |
|-------|-------|--------|----------------|
| #60 | SSH StrictHostKeyChecking disabled | S | A: Security Scripts |
| #62 | Unsafe rm -rf with potentially unset variable | S | A: Security Scripts |
| #81 | Race condition in inference lock file | M | A: Security Scripts |

### Dependencies
```
None - these are independent and can all be done in parallel
```

### Parallelization
All 3 issues can be worked on simultaneously by different developers:
- **Track A1:** #60, #62 (both in `scripts/`, same domain)
- **Track A2:** #81 (different script, independent)

### Blocking Relationship
```
Phase 0 ──blocks──> All subsequent phases
   │
   └── #60, #62, #81 must be merged before deployment
```

---

## Phase 1: Security & Stability

Security hardening that should be completed before production traffic.

### Issues

| Issue | Title | Effort | Parallel Group | Blocked By |
|-------|-------|--------|----------------|------------|
| #61 | DB credentials duplicated across 9 files | L | C: DB Credentials | None |
| #63 | Backup files without restrictive permissions | XS | A: Security Scripts | None |
| #89 | Cloud credentials not validated at startup | S | C: DB Credentials | None |

### Dependencies
```
#61 and #89 both touch credential handling but are independent files
```

### Parallelization
- **Track A:** #63 (quick script fix)
- **Track C:** #61 + #89 (credential-related, same developer recommended)

### Blocking Relationship
```
#61 (DB credentials) ──blocks──> #85 (Connection pool consolidation)
                                 (Must centralize credentials before
                                  refactoring connection patterns)
```

---

## Phase 2: Error Handling & Resilience

Critical for production stability - retry logic, connection handling, resource cleanup.

### Issues

| Issue | Title | Effort | Parallel Group | Blocked By |
|-------|-------|--------|----------------|------------|
| #64 | Cursor resource leaks in evaluation_job.py | S | B: Collector Resilience | None |
| #65 | Silent connection recovery failures in collectors | M | B: Collector Resilience | None |
| #66 | No retry logic for API failures in collectors | M | B: Collector Resilience | None |
| #67 | Generic exception handlers hiding bugs | M | H: Code Quality | None |
| #90 | Bare except clauses in transform_service.py | S | B: Collector Resilience | None |

### Dependencies
```
#65 (connection recovery) + #66 (retry logic) are related but independent
#87 (circuit breaker) in Phase 6 depends on #66 being done first
```

### Parallelization
- **Track B1:** #64, #90 (resource cleanup patterns)
- **Track B2:** #65, #66 (collector resilience - same files)
- **Track H:** #67 (training code, independent)

### Blocking Relationship
```
#66 (retry logic) ──blocks──> #87 (circuit breaker pattern)
                              (Add retries first, then circuit breaker)
```

---

## Phase 3: Configuration & Monitoring

Centralize configuration and improve observability. Large phase with many parallel opportunities.

### Issues

| Issue | Title | Effort | Parallel Group | Blocked By |
|-------|-------|--------|----------------|------------|
| #68 | Hardcoded server IP in 14+ locations | M | D: Configuration | None |
| #69 | Duplicate CatBoost parameters across 4 files | M | D: Configuration | None |
| #70 | Magic numbers for AUC confidence thresholds | S | D: Configuration | None |
| #71 | Duplicate tier thresholds in training files | S | D: Configuration | #69 |
| #72 | Missing data quality observability metrics | L | E: Monitoring | None |
| #73 | Print statements instead of structured logging | M | E: Monitoring | None |
| #74 | Health checks only verify metrics endpoint | M | E: Monitoring | #72 |
| #75 | Training pipeline lacks progress telemetry | M | E: Monitoring | #73 |

### Dependencies
```
#71 depends on #69 - centralize CatBoost params first, then tier thresholds
#74 depends on #72 - add data quality metrics, then enhance health checks
#75 depends on #73 - fix logging first, then add telemetry
```

### Parallelization
- **Track D1:** #68 (infrastructure config)
- **Track D2:** #69 → #71 (training config, sequential)
- **Track D3:** #70 (inference config, independent)
- **Track E1:** #72 → #74 (collector monitoring, sequential)
- **Track E2:** #73 → #75 (logging/telemetry, sequential)

### Blocking Relationship
```
#69 ──blocks──> #71
#72 ──blocks──> #74
#73 ──blocks──> #75
```

---

## Phase 4: Data Quality

Input validation and data integrity improvements.

### Issues

| Issue | Title | Effort | Parallel Group | Blocked By |
|-------|-------|--------|----------------|------------|
| #76 | Missing validation for price data in collectors | M | F: Data Quality | #72 |
| #77 | Training data validation gaps | M | F: Data Quality | None |
| #78 | Silent fallback to 0.0 for missing price data | S | F: Data Quality | #76 |
| #79 | News collector truncates data without logging | S | F: Data Quality | #73 |

### Dependencies
```
#76 should be done after #72 (need metrics to track validation failures)
#78 depends on #76 (fix validation first, then handle fallback)
#79 depends on #73 (need logging before adding log messages)
```

### Parallelization
- **Track F1:** #76 → #78 (price data validation, sequential)
- **Track F2:** #77 (training validation, independent)
- **Track F3:** #79 (news validation, after #73)

### Blocking Relationship
```
#72 ──blocks──> #76 ──blocks──> #78
#73 ──blocks──> #79
```

---

## Phase 5: Operations & DevOps

Deployment safety, backup verification, operational improvements.

### Issues

| Issue | Title | Effort | Parallel Group | Blocked By |
|-------|-------|--------|----------------|------------|
| #80 | No deployment rollback capability | L | G: Operations | #68 |
| #82 | No backup verification or restore testing | M | G: Operations | None |
| #83 | No resource limits in Docker containers | S | G: Operations | None |
| #84 | Shell scripts missing set -u | M | A: Security Scripts | Phase 0 |

### Dependencies
```
#80 depends on #68 - centralize server config before adding rollback
#84 should be done after Phase 0 scripts are stable
```

### Parallelization
- **Track G1:** #82, #83 (independent operational improvements)
- **Track G2:** #80 (after #68)
- **Track A3:** #84 (after Phase 0)

### Blocking Relationship
```
#68 ──blocks──> #80
Phase 0 ──blocks──> #84
```

---

## Phase 6: Code Quality & Performance

Technical debt and performance optimizations. Lower priority but improves maintainability.

### Issues

| Issue | Title | Effort | Parallel Group | Blocked By |
|-------|-------|--------|----------------|------------|
| #85 | Connection pool not consistently used | M | H: Code Quality | #61 |
| #86 | GPU memory management issues in training | M | I: Performance | None |
| #87 | Add circuit breaker pattern for API calls | M | B: Collector Resilience | #66 |
| #88 | Player count collector creates new DB connection | S | I: Performance | #85 |

### Dependencies
```
#85 depends on #61 - centralize credentials before refactoring connections
#87 depends on #66 - add retry logic before circuit breaker
#88 depends on #85 - standardize connection pool before fixing individual collectors
```

### Parallelization
- **Track H:** #85 → #88 (connection management, sequential)
- **Track B3:** #87 (after #66)
- **Track I:** #86 (GPU optimization, independent)

### Blocking Relationship
```
#61 ──blocks──> #85 ──blocks──> #88
#66 ──blocks──> #87
```

---

## Complete Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │         PHASE 0: CRITICAL               │
                    │    #60, #62, #81 (all parallel)         │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         PHASE 1: SECURITY               │
                    │  ┌─────┐  ┌─────┐  ┌─────┐              │
                    │  │ #63 │  │ #61 │  │ #89 │              │
                    │  └─────┘  └──┬──┘  └─────┘              │
                    └──────────────┼──────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        │                          │                              │
        ▼                          ▼                              │
┌───────────────────┐    ┌─────────────────────┐                  │
│   PHASE 2: ERROR  │    │  PHASE 6 (partial)  │                  │
│   #64, #65, #66   │    │       #85           │                  │
│   #67, #90        │    │        │            │                  │
└────────┬──────────┘    │        ▼            │                  │
         │               │       #88           │                  │
         │               └─────────────────────┘                  │
         │                                                        │
         ▼                                                        │
┌─────────────────────────────────────────────────────────────────┴─┐
│                    PHASE 3: CONFIG & MONITORING                   │
│  ┌─────┐  ┌─────┬─────┐  ┌─────┐  ┌─────┬─────┐  ┌─────┬─────┐   │
│  │ #68 │  │ #69 │ #71 │  │ #70 │  │ #72 │ #74 │  │ #73 │ #75 │   │
│  └──┬──┘  └──┬──┴──▲──┘  └─────┘  └──┬──┴──▲──┘  └──┬──┴──▲──┘   │
│     │        └─────┘                 └─────┘        └─────┘       │
└─────┼───────────────────────────────────┬───────────────┬────────┘
      │                                   │               │
      ▼                                   ▼               ▼
┌───────────────┐               ┌─────────────────────────────────┐
│ PHASE 5 (part)│               │       PHASE 4: DATA QUALITY     │
│     #80       │               │  ┌─────┬─────┐  ┌─────┐ ┌─────┐ │
└───────────────┘               │  │ #76 │ #78 │  │ #77 │ │ #79 │ │
                                │  └──┬──┴──▲──┘  └─────┘ └──▲──┘ │
                                │     └─────┘                │     │
                                └────────────────────────────┼─────┘
                                                             │
                                    (depends on #73) ────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 5: OPERATIONS                          │
│  ┌─────┐  ┌─────┐  ┌─────┐                                      │
│  │ #82 │  │ #83 │  │ #84 │  (+ #80 after #68)                   │
│  └─────┘  └─────┘  └─────┘                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 6: CODE QUALITY                        │
│  ┌─────┐  ┌─────┐                                               │
│  │ #86 │  │ #87 │  (after #66)                                  │
│  └─────┘  └─────┘                                               │
│  (+ #85, #88 after #61)                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Execution Order

### Week 1: Critical & Security (3 developers)

| Day | Dev 1 | Dev 2 | Dev 3 |
|-----|-------|-------|-------|
| 1 | #60 (SSH) | #62 (rm -rf) | #81 (lock file) |
| 1 | #63 (backup perms) | #84 (set -u) | #89 (credentials) |
| 2-3 | #61 (DB credentials - large) | #65 (connection recovery) | #66 (retry logic) |
| 3 | | #64 (cursor leaks) | #90 (bare except) |
| 4 | #67 (generic exceptions) | #87 (circuit breaker) | #85 (connection pool) |

### Week 2: Configuration & Monitoring (3 developers)

| Day | Dev 1 | Dev 2 | Dev 3 |
|-----|-------|-------|-------|
| 1 | #68 (hardcoded IPs) | #69 (CatBoost params) | #72 (data quality metrics) |
| 2 | #70 (magic numbers) | #71 (tier thresholds) | #73 (print → logging) |
| 3 | #80 (rollback) | #82 (backup verify) | #74 (health checks) |
| 4 | #83 (Docker limits) | #77 (training validation) | #75 (training telemetry) |

### Week 3: Data Quality & Performance (2 developers)

| Day | Dev 1 | Dev 2 |
|-----|-------|-------|
| 1 | #76 (price validation) | #86 (GPU memory) |
| 2 | #78 (silent fallback) | #88 (player count conn) |
| 3 | #79 (news truncation) | Buffer/review |

---

## Quick Reference: What Blocks What

| If working on... | Must complete first... |
|------------------|------------------------|
| #71 (tier thresholds) | #69 (CatBoost params) |
| #74 (health checks) | #72 (data quality metrics) |
| #75 (training telemetry) | #73 (print → logging) |
| #76 (price validation) | #72 (data quality metrics) |
| #78 (silent fallback) | #76 (price validation) |
| #79 (news truncation) | #73 (print → logging) |
| #80 (rollback) | #68 (hardcoded IPs) |
| #84 (set -u) | Phase 0 complete |
| #85 (connection pool) | #61 (DB credentials) |
| #87 (circuit breaker) | #66 (retry logic) |
| #88 (player count) | #85 (connection pool) |

---

## Issues With No Dependencies (Can Start Anytime)

These issues have no blockers and can be picked up independently:

- #60 - SSH StrictHostKeyChecking
- #62 - Unsafe rm -rf
- #63 - Backup permissions
- #64 - Cursor leaks
- #65 - Connection recovery
- #66 - Retry logic
- #67 - Generic exceptions
- #68 - Hardcoded IPs
- #69 - CatBoost params
- #70 - Magic numbers
- #72 - Data quality metrics
- #73 - Print → logging
- #77 - Training validation
- #81 - Lock file race
- #82 - Backup verification
- #83 - Docker limits
- #86 - GPU memory
- #89 - Cloud credentials
- #90 - Bare except

**Total independent issues:** 19 out of 31 (61%)

---

## Success Criteria

### Phase 0 Complete
- [ ] All scripts use `set -euo pipefail`
- [ ] No unsafe `rm -rf` with unset variables
- [ ] SSH connections verify host keys
- [ ] Lock files use atomic locking

### Phase 1 Complete
- [ ] All modules import credentials from `db_utils.py`
- [ ] Backup files have 600 permissions
- [ ] Cloud credentials validated at startup

### Phase 2 Complete
- [ ] All API calls have retry logic
- [ ] Connection recovery is logged and retried
- [ ] No bare `except` clauses
- [ ] All cursors use context managers

### Full Hardening Complete
- [ ] All 31 issues closed
- [ ] No critical/high priority issues open
- [ ] All tests passing
- [ ] Deployment successful with rollback capability
