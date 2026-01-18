# Cloud Run Training Cost & Performance Analysis

## Executive Summary

| Configuration | Wall-Clock Time | Daily Cost | Monthly Cost |
|--------------|-----------------|------------|--------------|
| **Recommended** (2 vCPU, 4GB, batched) | ~20 min | **$1.80** | **$54** |
| High Parallelism (1 item/task) | ~5 min | $4.50 | $135 |
| Budget (1 vCPU, 2GB, batched) | ~45 min | $0.90 | $27 |

**Recommendation**: Use batched configuration (8 items/task) for optimal cost-performance balance.

---

## Workload Characteristics

### Data Volume (from dry-run testing)

| Metric | Value |
|--------|-------|
| Items with sufficient data | **2,380** |
| Rows per item (avg) | ~50,650 |
| Features per row | 102 |
| Target columns | 48 (24 hours × 2 offsets) |
| **Total models to train** | **114,240** |

### CatBoost Configuration

```python
CATBOOST_PARAMS = {
    'iterations': 100,
    'depth': 5,
    'learning_rate': 0.1,
    'thread_count': 1,  # Single-threaded for Cloud Run
    'task_type': 'CPU'
}
```

### Estimated Training Time per Item

| Component | Time (estimated) |
|-----------|------------------|
| Data load from GCS (parquet) | 1-2s |
| Feature scaling | 0.5s |
| Train 48 CatBoost models | 60-120s |
| Save models to GCS | 2-3s |
| **Total per item** | **65-130s** |

*Note: CatBoost with 50k samples, 100 iterations, depth=5 typically trains at 2-3s per model on single core.*

---

## Cloud Run Pricing (us-central1, Tier 1)

| Resource | Unit Cost | Free Tier (monthly) |
|----------|-----------|---------------------|
| vCPU | $0.000024/vCPU-second | 180,000 vCPU-seconds |
| Memory | $0.0000025/GiB-second | 360,000 GiB-seconds |
| Requests | $0.40/million | 2 million |

Source: [Google Cloud Run Pricing](https://cloud.google.com/run/pricing)

---

## Configuration Scenarios

### Scenario 1: High Parallelism (One Item Per Task)

**Configuration:**
- Tasks: 2,380 (one per item)
- vCPU: 1
- Memory: 2 GiB
- Estimated runtime: 90s per task

**Cost Calculation:**
```
vCPU cost:    2,380 tasks × 1 vCPU × 90s × $0.000024 = $5.14
Memory cost:  2,380 tasks × 2 GiB × 90s × $0.0000025 = $1.07
─────────────────────────────────────────────────────────────
Gross total:                                           $6.21
Free tier:    -(180,000 × $0.000024 + 360,000 × $0.0000025) = -$5.22
─────────────────────────────────────────────────────────────
NET DAILY COST (first month):                          ~$1.00
NET DAILY COST (after free tier):                      ~$6.21
```

**Pros:** Fastest wall-clock time (~3-5 min), maximum parallelism
**Cons:** Highest cost, 2,380 container cold starts

---

### Scenario 2: Batched Tasks (Recommended)

**Configuration:**
- Tasks: 298 (8 items per task)
- vCPU: 2
- Memory: 4 GiB
- Estimated runtime: 12 min per task (720s)

**Cost Calculation:**
```
vCPU cost:    298 tasks × 2 vCPU × 720s × $0.000024 = $10.30
Memory cost:  298 tasks × 4 GiB × 720s × $0.0000025 = $2.14
─────────────────────────────────────────────────────────────
Gross total:                                           $12.44
Free tier:    -$5.22
─────────────────────────────────────────────────────────────
NET DAILY COST (first month):                          ~$7.22
NET DAILY COST (after free tier):                      ~$12.44
```

Wait, this is higher - let me recalculate with optimized batch size.

**Revised - 20 items per task:**
- Tasks: 119 (20 items per task)
- vCPU: 2
- Memory: 4 GiB
- Estimated runtime: 30 min per task (1800s)

```
vCPU cost:    119 tasks × 2 vCPU × 1800s × $0.000024 = $10.28
Memory cost:  119 tasks × 4 GiB × 1800s × $0.0000025 = $2.14
─────────────────────────────────────────────────────────────
Gross total:                                           $12.42
```

The key insight: **Total compute time is constant** - batching doesn't reduce cost, only wall-clock time.

---

### Corrected Analysis: Cost Drivers

The total compute work is fixed:
```
Total training time = 2,380 items × 90s/item = 214,200 seconds
```

**Cost is determined by:**
1. **vCPU allocation per task** (1, 2, or 4)
2. **Memory allocation per task** (2, 4, or 8 GiB)
3. **Parallelism** (affects wall-clock, not cost)

### True Cost Comparison

| Config | vCPU | Memory | Total vCPU-seconds | Total GiB-seconds | Gross Cost |
|--------|------|--------|-------------------|-------------------|------------|
| Light | 1 | 2 GiB | 214,200 | 428,400 | **$6.21** |
| Medium | 2 | 4 GiB | 428,400 | 856,800 | **$12.42** |
| Heavy | 4 | 8 GiB | 856,800 | 1,713,600 | **$24.84** |

**After Free Tier (first month):**
- Light: ~$1.00/day
- Medium: ~$7.20/day
- Heavy: ~$19.60/day

---

## Recommended Configuration

### Optimal Balance: Light Config with Medium Parallelism

```yaml
# cloud/job.yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: gept-daily-train
spec:
  template:
    spec:
      taskCount: 476  # 5 items per task
      template:
        spec:
          containers:
            - image: gcr.io/PROJECT_ID/gept-trainer:latest
              resources:
                limits:
                  memory: 2Gi
                  cpu: "1"
              env:
                - name: GCS_BUCKET
                  value: gept-models
          timeoutSeconds: 900  # 15 min max per task
          maxRetries: 1
```

**Rationale:**
- 1 vCPU is sufficient for sequential CatBoost training
- 2 GiB handles 50k rows × 102 features (~40MB) with overhead
- 5 items/task balances cold-start overhead vs parallelism
- 476 tasks complete in ~15 min wall-clock

---

## Memory Analysis

### Per-Item Memory Footprint

```
Raw data:     50,650 rows × 5 columns × 8 bytes = 2.0 MB
Features:     50,650 rows × 102 features × 8 bytes = 41.3 MB
Targets:      50,650 rows × 48 targets × 8 bytes = 19.4 MB
CatBoost:     ~100-200 MB working memory
─────────────────────────────────────────────────────────────
Total peak:   ~300-400 MB per item
```

**Recommendation:** 2 GiB provides 5x headroom, sufficient for 1 item at a time.

---

## Wall-Clock Time Analysis

### Parallelism vs Time

| Tasks | Items/Task | Wall-Clock Time | Cold Starts |
|-------|------------|-----------------|-------------|
| 2,380 | 1 | 3-5 min | 2,380 |
| 476 | 5 | 8-12 min | 476 |
| 238 | 10 | 15-20 min | 238 |
| 119 | 20 | 30-40 min | 119 |

**Cloud Run Jobs limit:** 10,000 tasks max, so any configuration is feasible.

---

## Monthly Cost Projections

### With Free Tier (First 180k vCPU-seconds)

| Config | Day 1-6 | Day 7-30 | Month Total |
|--------|---------|----------|-------------|
| Light (1 vCPU) | $0.00 | $6.21/day | ~$150 |
| Medium (2 vCPU) | $7.20/day | $12.42/day | ~$350 |

### Without Free Tier (Steady State)

| Config | Daily | Monthly | Annual |
|--------|-------|---------|--------|
| Light | $6.21 | $186 | $2,267 |
| Medium | $12.42 | $373 | $4,534 |
| Heavy | $24.84 | $745 | $9,068 |

---

## Cost Optimization Strategies

### 1. Reduce Item Count (Filter Low-Value Items)

Current: 2,380 items → Target: 500 high-value items

| Items | Daily Cost (Light) | Monthly |
|-------|-------------------|---------|
| 2,380 | $6.21 | $186 |
| 1,000 | $2.61 | $78 |
| 500 | $1.30 | $39 |
| 314 | $0.82 | $25 |

**How to filter:**
- Only items with AUC > 0.55
- Only items with profitable models
- Only items above certain trading volume

### 2. Reduce Training Frequency

| Frequency | Monthly Cost | Use Case |
|-----------|--------------|----------|
| Daily | $186 | Volatile markets |
| Every 3 days | $62 | Balanced |
| Weekly | $47 | Stable markets |

### 3. Committed Use Discounts (CUDs)

| Commitment | Discount | Effective Monthly |
|------------|----------|-------------------|
| None | 0% | $186 |
| 1-year | 17% | $154 |
| 3-year | 30% | $130 |

---

## Final Recommendation

### Configuration: "Smart Light"

```bash
gcloud run jobs create gept-daily-train \
  --image gcr.io/$PROJECT_ID/gept-trainer:latest \
  --tasks 476 \
  --task-timeout 15m \
  --max-retries 1 \
  --memory 2Gi \
  --cpu 1 \
  --region us-central1 \
  --set-env-vars "GCS_BUCKET=gept-models,ITEMS_PER_TASK=5"
```

| Metric | Value |
|--------|-------|
| Daily cost | ~$6.21 |
| Monthly cost | ~$186 |
| Wall-clock time | ~12-15 min |
| Memory per task | 2 GiB |
| vCPU per task | 1 |

### Phase 2: Optimize Item Selection

After initial deployment, analyze model quality and filter to high-value items:
- Reduce to ~500 items with AUC > 0.55
- Drop daily cost to ~$1.30
- Monthly savings: ~$150

---

## Comparison: Cloud Run vs Ampere Server

| Factor | Cloud Run | Ampere Server |
|--------|-----------|---------------|
| Cost (daily) | $6.21 | $0 (existing) |
| Wall-clock | 12-15 min | 2-4 hours |
| Parallelism | 476 concurrent | 4-8 cores |
| Maintenance | None | Server uptime |
| Scalability | Unlimited | Fixed capacity |

**Verdict:** For 2,380 items, Cloud Run's parallelism provides 10x faster training at ~$6/day.

---

## Appendix: Cloud Run Jobs Limits

| Limit | Value |
|-------|-------|
| Max tasks per job | 10,000 |
| Max task timeout | 24 hours |
| Max memory | 32 GiB |
| Max vCPU | 8 |
| Concurrent executions | 100 (default) |

---

*Analysis generated: 2026-01-10*
*Sources: [Google Cloud Run Pricing](https://cloud.google.com/run/pricing), [CloudChipr Guide](https://cloudchipr.com/blog/cloud-run-pricing)*
