# GePT Infrastructure Audit Report
**Date:** 2026-01-18
**Auditor:** Claude Code

---

## Executive Summary

### Overall System Health: **GOOD** (with critical items requiring attention)

GePT is a well-architected ML system for OSRS Grand Exchange price prediction. The system is functional and serving users through gept.gg, with continuous model training and real-time inference operating successfully.

### Critical Issues Requiring Immediate Action
1. **P0 - Disk Space Critical**: Ampere server at 82% disk usage (8.4GB remaining)
2. **P1 - Security**: Hardcoded default password in `cloud/prepare_training_data.py`
3. **P1 - Model Sync Broken**: Hydra cannot sync models to Ampere (SSH key issue #145)

### Top 5 Highest-Impact Improvements
1. **Migrate to larger storage** - Prevents data loss and system failure
2. **Remove GCP legacy code** - 1,617 lines of unused code
3. **Consolidate predictor implementations** - 4 redundant predictor classes
4. **Fix model sync pipeline** - Enable automatic model deployment
5. **Add rate limiting to API** - Security and cost protection

---

## 1. Production vs Repository Reconciliation

### Repository Inventory
| Repository | Purpose | Status | Last Updated |
|------------|---------|--------|--------------|
| `gept-foundations` | ML training & inference core | **Active** | 2026-01-18 |
| `gept-gg` | Frontend (Astro/Vercel) | **Active** | 2026-01-18 |
| `gept-discord-bot` | Discord bot (TypeScript) | **Active** | 2026-01-15 |
| `gept-recommendation-engine` | Recommendation API | **Active** | 2026-01-18 |
| `osrs-data-collection` | Data collectors | **Unclear** | 2026-01-06 |
| `osrs-reddit-collector` | Reddit data | Archive candidate | 2026-01-05 |
| `osrs-ge-trader` | Legacy | Archive candidate | 2026-01-01 |
| `ML-Trader` | Legacy | Archive candidate | 2025-12-30 |
| `GE-Oracle-ML` | Legacy | Archive candidate | 2025-12-30 |
| `osrs_trading_assistant` | Legacy | Archive candidate | 2025-08-10 |

### Production Services on Ampere (150.136.170.128)

**Docker Containers:**
| Container | Status | Port | Purpose |
|-----------|--------|------|---------|
| gept-engine | Up (healthy) | 8000 | Recommendation API |
| osrs-ge-collector | Up 6 days | 9100 | 5-min price collector |
| osrs-news-collector | Up 6 days | 9102 | News collector |
| osrs-hourly-collector | Up 6 days | 9101 | Hourly data |
| osrs-dashboard | Up 7 days | 8080 | Status dashboard |
| gept-grafana | Up 7 days | 3001 | Monitoring |
| gept-prometheus | Up 7 days | 9090 | Metrics |
| osrs-latest-1m | Up 7 days | 9103 | 1-min ticks |
| metabase | Up 11 days | - | BI tool |

**Systemd Services:**
| Service | Status | Purpose |
|---------|--------|---------|
| gept-bot.service | **Running** | Discord bot |
| player_count.service | Running | Player count collector |
| item_updater.service | Running | Item metadata sync |

**Cron Jobs:**
- `0 2 * * *` - Daily database backup
- `*/5 * * * *` - Fallback inference

### Production Services on Hydra (10.0.0.146)

**Running Processes:**
- `continuous_scheduler --bootstrap` - Multi-target model training (52/746 items completed)

**Cron Jobs:**
- `*/5 * * * *` - Primary inference with local cache

### Key Findings

**Discord Bot Status:** The Discord bot is **NOT dead code** - it runs as a separate TypeScript application on Ampere via systemd. The session cache shows 0/5000 users, indicating very low current usage but the system is operational.

**Code Safe to Delete from gept-foundations:**
1. `cloud/gcp_train.py` (357 lines) - GCP Cloud Run training
2. `cloud/gcp_train_catboost.py` (260 lines) - GCP CatBoost variant
3. `cloud/gcp_train_from_gcs.py` (593 lines) - GCS-based training
4. `cloud/gcp_train_multimodel.py` (407 lines) - Multi-model GCP
5. `cloud/prepare_training_data.py` (partial - GCS upload code)

**Total Dead Code:** ~1,617 lines in GCP training scripts

---

## 2. Architecture Diagram

```
                                    ┌─────────────────────────────────┐
                                    │          USERS                  │
                                    │   gept.gg (Vercel/Astro)       │
                                    └───────────────┬─────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AMPERE SERVER (Oracle ARM)                           │
│                         150.136.170.128                                      │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  gept-engine    │  │  gept-bot       │  │  Data Collectors            │ │
│  │  (FastAPI:8000) │  │  (Discord)      │  │  - 5min (9100)              │ │
│  │                 │  │                 │  │  - 1min (9103)              │ │
│  │  /api/v1/*      │  │  Session: 0     │  │  - hourly (9101)            │ │
│  │  predictions    │  │  users          │  │  - news (9102)              │ │
│  └────────┬────────┘  └─────────────────┘  └───────────┬─────────────────┘ │
│           │                                            │                    │
│           └──────────────────┬─────────────────────────┘                    │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    PostgreSQL/TimescaleDB                             │ │
│  │  price_data_5min: 426M rows (10GB+)   |   predictions: 54M rows      │ │
│  │  items: 4,500    |   model_registry   |   inference_status           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  Prometheus     │  │  Grafana        │  │  Metabase       │            │
│  │  (9090)         │  │  (3001)         │  │  (BI)           │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ SSH Tunnel (models sync - BROKEN)
                                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                         HYDRA SERVER (Local)                                 │
│                         10.0.0.146                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  continuous_scheduler.py                                                ││
│  │  - Multi-target CatBoost training (108 targets/item)                   ││
│  │  - RTX 3060 GPU (12GB VRAM)                                            ││
│  │  - Progress: 52/746 items                                              ││
│  │  - AUC range: 0.85-0.99                                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  run_inference_hydra.sh (cron */5)                                     ││
│  │  - Primary inference runner                                             ││
│  │  - Uses local Parquet cache (373MB)                                    ││
│  │  - Fast mode: 0.10s inference time                                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                  │
│  │  data_cache/    │  │  models/        │                                  │
│  │  (Parquet)      │  │  (CatBoost)     │                                  │
│  └─────────────────┘  └─────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware & Resource Utilization

### Ampere Server (Oracle Cloud ARM)
| Resource | Spec | Current Usage | Status |
|----------|------|---------------|--------|
| CPU | 4 ARM cores (Neoverse-N1) | Low | OK |
| Memory | 23GB | 2.4GB (10%) | OK |
| GPU | None | N/A | N/A |
| Disk | 45GB | 37GB (82%) | **CRITICAL** |
| Cost | Free tier | $0/month | OK |

### Hydra Server (Local)
| Resource | Spec | Current Usage | Status |
|----------|------|---------------|--------|
| CPU | Intel i7-10700F (16 threads) | Low (~5%) | OK |
| Memory | 32GB | 1.4GB (4%) | OK |
| GPU | RTX 3060 12GB | 104MB (training) | OK |
| Disk | 98GB | 16GB (17%) | OK |
| Cost | Electricity only | ~$15/month | OK |

### Resource Optimization Opportunities

1. **Ampere Disk Critical** - Need to migrate to 100GB block volume (Issue #141)
2. **Hydra Underutilized** - GPU at <1% during inference, memory at 4%
3. **Training Efficiency** - Current: ~30s/item, could be faster with batch training

---

## 4. Model Training Pipeline Assessment

### Current Pipeline
```
Data Sync → Feature Computation → Target Generation → CatBoost Training → Validation → Deployment
  │              │                      │                   │                │            │
  │              │                      │                   │                │            │
  ▼              ▼                      ▼                   ▼                ▼            ▼
Parquet      FeatureEngine       TargetEngine          GPU Training      AUC Check    Registry
(373MB)      (474 lines)         (700 lines)           (1487 lines)      (>0.7)       (DB)
```

### Performance Benchmarks
| Stage | Time | Notes |
|-------|------|-------|
| Data Load | 0.3s | From local Parquet cache |
| Feature Compute | 1.7s | 52 items, 8 workers |
| Inference | 0.10s | Fast mode, single pass |
| DB Write | 0.44s | COPY protocol |
| **Total Cycle** | **3.9s** | Well under 5-min window |

| Training | Time | Notes |
|----------|------|-------|
| Per Item | ~30s | Including validation |
| Full Bootstrap | ~6-8 hours | 746 items |
| Weekly Refresh | ~3-4 hours | 200 items/day quota |

### Quality Metrics
- Model AUC Range: 0.85-0.99 (excellent)
- Models Deployed: 418 COMPLETED, 4 TRAINING
- Models Pending: 416 (current bootstrap)
- Failed: 8 (1.9% failure rate)

### Complexity Issues

**continuous_scheduler.py: 1,487 lines** - This is the largest file and handles:
- Item prioritization
- GPU training coordination
- Model validation
- Registry management
- Drift detection
- Sync to Ampere

Consider splitting into:
- `scheduler.py` - Core scheduling
- `gpu_trainer.py` - Training logic
- `model_publisher.py` - Validation & deployment

---

## 5. Data Handling Assessment

### Database Tables (Top 15 by Size)
| Table | Size | Rows | Purpose |
|-------|------|------|---------|
| prices_latest_1m | 10 GB | 63M | 1-minute tick data |
| _hyper_3_* chunks | 10+ GB | 47M+ | TimescaleDB price chunks |
| prices_latest | 3.5 GB | 20M | Latest prices |
| prices_1h | 288 MB | 1.8M | Hourly aggregates |
| predictions | ~1 GB | 54M | Model predictions |

### Data Quality
- ✅ TimescaleDB compression enabled
- ✅ Proper indexing on time columns
- ✅ Data validation in collectors
- ⚠️ No explicit data retention policy
- ⚠️ prices_latest_1m growing unbounded (63M rows)

### Recommendations
1. **Implement retention policy** - Drop data older than 6 months
2. **Add monitoring** - Alert on data gaps
3. **Optimize prices_latest_1m** - Consider downsampling old data

---

## 6. Inference System Assessment

### Current Architecture
- Primary: Hydra (local, with RTX 3060)
- Fallback: Ampere (ARM, CPU-only)
- Refresh: Every 5 minutes
- Latency: <4 seconds end-to-end

### Performance
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Inference Latency | 0.10s | <1s | ✅ Excellent |
| Total Cycle | 3.9s | <60s | ✅ Excellent |
| Predictions/Cycle | 1,728 | - | OK |
| Items Covered | 53 | 746 | ⚠️ Bootstrap in progress |

### Reliability
- ✅ Fallback mechanism (Ampere checks freshness)
- ✅ Status tracking in inference_status table
- ✅ Error logging
- ⚠️ No alerting on failures
- ⚠️ Model sync broken (Issue #145)

---

## 7. Codebase Audit

### Dead Code Inventory

**GCP Legacy Code (Safe to Delete):**
| File | Lines | Reason |
|------|-------|--------|
| cloud/gcp_train.py | 357 | Replaced by RunPod/local training |
| cloud/gcp_train_catboost.py | 260 | Replaced by train_runpod_multitarget.py |
| cloud/gcp_train_from_gcs.py | 593 | No longer using GCS |
| cloud/gcp_train_multimodel.py | 407 | Replaced by continuous_scheduler.py |
| **Total** | **1,617** | |

**Redundant Predictor Implementations:**
| File | Lines | Recommendation |
|------|-------|----------------|
| batch_predictor.py | 666 | Archive (sklearn-based) |
| batch_predictor_fast.py | 552 | Archive (legacy CatBoost) |
| batch_predictor_pertarget.py | ? | Keep for per-target models |
| batch_predictor_multitarget.py | 1,250 | **Primary - Keep** |

### Complexity Hotspots
| File | Lines | Concern |
|------|-------|---------|
| continuous_scheduler.py | 1,487 | Too many responsibilities |
| batch_predictor_multitarget.py | 1,250 | Acceptable for core inference |
| item_selector.py | 1,240 | Complex item selection logic |
| trainer.py | 860 | Legacy training code |
| priority_queue.py | 857 | Acceptable complexity |

### Test Coverage
- 7 test files, 1,934 lines of tests
- Key areas tested: features, targets, calibration, sequential fills
- ⚠️ Missing: API tests, integration tests, inference tests

---

## 8. Security Audit

### Critical Issues

**P1 - Hardcoded Password:**
```python
# cloud/prepare_training_data.py:44
'password': os.environ.get('DB_PASS', 'osrs_price_data_2024')
```
**Risk:** Exposed in version control, easily guessable
**Fix:** Remove default, require env var

### Authentication & Authorization
| Component | Auth Method | Status |
|-----------|-------------|--------|
| API (gept-engine) | None (public) | ⚠️ Rate limiting needed |
| Database | Password | ✅ Env vars |
| SSH | Key-based | ✅ Good |
| Grafana | Password | ✅ Env vars |

### Secrets Management
| Secret | Storage | Status |
|--------|---------|--------|
| DB_PASS | .env (git-ignored) | ✅ OK |
| SSH Keys | .secrets/ (git-ignored) | ✅ OK |
| GCP Service Account | .secrets/*.json | ✅ OK |
| Discord Token | .env on Ampere | ✅ OK |

### Network Security
- ⚠️ API publicly accessible without rate limiting
- ⚠️ Dashboard publicly accessible (8080)
- ✅ Database not exposed externally
- ✅ CORS configured properly

### Recommendations
1. **Add rate limiting** to API endpoints
2. **Restrict dashboard** to VPN/internal only
3. **Remove hardcoded password** from prepare_training_data.py
4. **Add authentication** for admin endpoints

---

## 9. Scalability Assessment

### Current Scaling Limits
| Component | Current | 10x Users | 100x Users |
|-----------|---------|-----------|------------|
| API Requests | ~10/min | 100/min | 1000/min |
| Predictions/Cycle | 1,728 | OK | OK |
| DB Connections | 5-10 | Need pool | Need pool |
| Storage | 37GB/45GB | **FAIL** | **FAIL** |

### Cost Scaling
| Scale | Ampere | Hydra | Total |
|-------|--------|-------|-------|
| Current | $0 | ~$15 | ~$15/mo |
| 10x Users | $0 | ~$15 | ~$15/mo |
| 100x Users | ~$50 (upgrade) | ~$15 | ~$65/mo |

### Bottlenecks
1. **Disk Space** - Will fail before user growth becomes issue
2. **Database Connections** - No connection pooling in API
3. **Single API Instance** - No horizontal scaling

### Scalability Roadmap
1. **Immediate:** Expand Ampere storage (Issue #141)
2. **Short-term:** Add connection pooling to API
3. **Medium-term:** Add Redis caching for predictions
4. **Long-term:** Consider managed database (Neon, Supabase)

---

## 10. Prioritized Action Plan

### P0 - Critical (Do Immediately)
| Issue | Impact | Effort | Action |
|-------|--------|--------|--------|
| Disk space 82% | System failure | M | Expand to 100GB volume |
| Hardcoded password | Security | S | Remove default from code |

### P1 - High Priority (This Week)
| Issue | Impact | Effort | Action |
|-------|--------|--------|--------|
| Model sync broken | No auto-deploy | S | Fix SSH key config (#145) |
| API rate limiting | Security/Cost | M | Add slowapi middleware |
| Data retention | Storage growth | M | Implement 6-month policy |

### P2 - Medium Priority (This Month)
| Issue | Impact | Effort | Action |
|-------|--------|--------|--------|
| GCP dead code | Complexity | S | Delete 1,617 lines |
| Redundant predictors | Confusion | M | Archive 3 old implementations |
| continuous_scheduler size | Maintainability | L | Split into 3 modules |
| Missing API tests | Quality | M | Add test suite |

### P3 - Low Priority (Backlog)
| Issue | Impact | Effort | Action |
|-------|--------|--------|--------|
| Archive legacy repos | Organization | S | Archive 4 old repos |
| Dashboard access | Security | S | Restrict to internal |
| prices_latest_1m growth | Storage | M | Downsample old data |
| Connection pooling | Scalability | M | Add to API |

---

## 11. GitHub Issues to Create

Based on this audit, the following issues should be created:

1. **[P0] critical-security: Remove hardcoded default password in prepare_training_data.py**
2. **[P0] reliability: Expand Ampere storage to 100GB (82% full)**
3. **[P1] security: Add rate limiting to gept-engine API**
4. **[P2] dead-code: Remove GCP training legacy code (1,617 lines)**
5. **[P2] complexity: Archive legacy predictor implementations**
6. **[P2] complexity: Split continuous_scheduler.py into smaller modules**
7. **[P2] quality: Add API integration tests**
8. **[P2] tech-debt: Implement data retention policy for prices_latest_1m**
9. **[P3] security: Restrict dashboard access to internal network**
10. **[P3] scalability: Add database connection pooling to API**

---

## Appendix A: Service Inventory

### Active Services (Keep)
- gept-engine (FastAPI)
- gept-bot (Discord)
- continuous_scheduler (Training)
- All collectors (5min, 1min, hourly, news)
- Prometheus/Grafana monitoring

### To Archive
- GCP training scripts
- Legacy predictor implementations
- osrs-ge-trader repository
- ML-Trader repository
- GE-Oracle-ML repository

### To Monitor
- Discord bot usage (currently 0 active users)
- osrs-data-collection repository status

---

## Appendix B: File Deletion Checklist

**Safe to Delete:**
- [ ] cloud/gcp_train.py
- [ ] cloud/gcp_train_catboost.py
- [ ] cloud/gcp_train_from_gcs.py
- [ ] cloud/gcp_train_multimodel.py

**Archive (Move to archive/ directory):**
- [ ] src/batch_predictor.py
- [ ] src/batch_predictor_fast.py

**Keep:**
- [x] src/batch_predictor_multitarget.py (primary)
- [x] src/batch_predictor_pertarget.py (for per-target models)
- [x] cloud/train_runpod_multitarget.py (current training)

---

*Report generated by Claude Code infrastructure audit*
