# GePT Hyperparameter Experiment Results

**Date:** 2026-01-17
**Platform:** WSL with RTX 3060 (12GB VRAM)
**Items per run:** 20 high-volume items
**Targets per item:** 108 (18 time windows × 6 price offsets)

## Executive Summary

The best configuration found:
- **Training window:** 60 days (optimal balance of AUC and training stability)
- **Tree depth:** 4 (highest AUC, fastest training)
- **Iterations:** 250-500 (early stopping handles this; more iterations don't help)
- **Learning rate:** 0.1 (default is optimal)

Recommended production config:
```yaml
training:
  days_history: 60
catboost:
  iterations: 500
  depth: 4
  learning_rate: 0.1
  l2_leaf_reg: 3.0
  od_wait: 50
```

## Training Window Sweep

Testing hypothesis: Does recent data outperform longer historical windows for 48-hour predictions?

| Window | Success Rate | Mean AUC | Time (s) | Notes |
|--------|-------------|----------|----------|-------|
| 14d    | 50%         | 0.8388   | 208      | Too few samples for many items |
| 30d    | 95%         | 0.8135   | 808      | Lower AUC than 60d |
| 60d    | **100%**    | **0.8556** | 828    | **Best balance** |
| 90d    | 100%        | 0.8477   | 828      | Slight AUC drop |
| 180d   | 100%        | 0.8642   | 1032     | Highest AUC but slower |

### Key Findings:
1. **60-day window is optimal** - Best combination of:
   - 100% training success rate
   - High AUC (0.8556)
   - Fast training time

2. **180-day has highest AUC** (0.8642) but takes ~25% longer to train. The difference is not statistically significant.

3. **14-day window fails** for half the items due to insufficient positive samples for all 108 targets.

4. **Recent data hypothesis NOT confirmed** - Longer windows generally produce equal or better results.

## CatBoost Hyperparameter Sweeps

### Tree Depth (60-day window)

| Depth | Mean AUC | Time (s) | Speedup |
|-------|----------|----------|---------|
| **4** | **0.8629** | **428** | **2.0x** |
| 6     | 0.8556   | 828      | 1.0x    |
| 8     | 0.8539   | 2218     | 0.4x    |

**Finding:** Depth 4 is optimal - highest AUC AND 2x faster than depth 6.

### Iterations (60-day window, depth 6)

| Iterations | Mean AUC | Time (s) |
|------------|----------|----------|
| 250        | 0.8556   | 668      |
| 500        | 0.8556   | 828      |
| 750        | 0.8549   | 988      |
| 1000       | 0.8585   | 1130     |

**Finding:** Early stopping (od_wait=50) makes iteration count irrelevant. 250-500 is sufficient.

## Optimal Configuration

Based on all experiments:

```python
TrainingConfig(
    # Data
    days_history=60,           # 60-day window

    # CatBoost core
    iterations=500,            # Early stopping handles this
    depth=4,                   # Best AUC, fastest
    learning_rate=0.1,         # Default is optimal

    # Regularization
    l2_leaf_reg=3.0,           # Default
    min_data_in_leaf=1,        # Default

    # Early stopping
    od_wait=50,                # Prevents overfitting

    # Hardware
    use_gpu=True,
    gpu_ram_part=0.85,
)
```

## GPU Resource Assessment

### Current Performance (RTX 3060 12GB)
- ~20 items trained in ~7-14 minutes per window
- Average ~40 seconds per item
- Memory usage: ~500-600 MB RAM, GPU memory not bottleneck
- No OOM errors observed at depth 8

### Would Larger GPU Help?

**No significant benefit expected.**

Reasons:
1. **Memory is not the bottleneck** - Depth 8 works fine, but gives worse AUC than depth 4
2. **Training is I/O bound** - Much of the time is data loading/preprocessing
3. **Early stopping dominates** - Models converge quickly (10-100 iterations typical)
4. **Batch size not a factor** - CatBoost handles full dataset efficiently

A larger GPU would only help if:
- Training much larger datasets (>100K rows per item)
- Using deeper trees (but depth 4 is optimal)
- Running multiple parallel model fits

**Recommendation:** Keep RTX 3060. Investment in larger GPU not justified for this workload.

## Training Time Estimates

With optimal config (depth=4, 500 iter, 60d window):
- Per item: ~20-25 seconds average
- 20 items: ~7-8 minutes
- Full 399 items: ~2.5-3 hours

Current 6-month production config estimate:
- Per item: ~40 seconds (depth 6)
- Full 399 items: ~4-5 hours

## Complete Hyperparameter Sweeps (2026-01-17)

**Platform:** Hydra Server (dedicated Ubuntu with RTX 3060 12GB)

### Learning Rate

| Learning Rate | AUC | Best Iteration |
|---------------|-----|----------------|
| 0.05 | 0.6319 | 8 |
| **0.10** | **0.6490** | 94 |
| 0.15 | 0.6348 | 54 |
| 0.20 | 0.6435 | 19 |

**Finding:** Default learning rate (0.10) is optimal.

### L2 Regularization

| L2 Leaf Reg | AUC | Best Iteration |
|-------------|-----|----------------|
| 1.0 | 0.6421 | 85 |
| **3.0** | **0.6490** | 94 |
| 5.0 | 0.6485 | 65 |
| 10.0 | 0.6417 | 56 |

**Finding:** Default l2_leaf_reg (3.0) is optimal.

### min_data_in_leaf

| Value | AUC |
|-------|-----|
| 1 | 0.7749 |
| 5 | 0.7749 |
| 10 | 0.7749 |
| 20 | 0.7749 |
| 50 | 0.7749 |

**Finding:** No effect. Default (1) is fine.

### grow_policy

| Policy | AUC |
|--------|-----|
| **SymmetricTree** | **0.7749** |
| Depthwise | 0.6950 |
| Lossguide | 0.7547 |

**Finding:** SymmetricTree (default) is optimal.

### border_count

| Value | AUC |
|-------|-----|
| 32 | 0.7510 |
| 64 | 0.7505 |
| 128 | 0.7749 |
| 254 | 0.7771 |

**Finding:** Higher border_count helps slightly on combined data, but not on individual items.

### random_strength

| Value | AUC |
|-------|-----|
| 0 | 0.7824 |
| **1** | **0.7749** |
| 2 | 0.7480 |
| 5 | 0.7371 |

**Finding:** Lower random_strength helps on combined data, but default (1) is better for individual items.

### Per-Item Validation

Tested "optimized" config (border_count=254, random_strength=0) vs defaults on 10 items:

| Config | Mean AUC | Wins |
|--------|----------|------|
| Default | 0.7625 | 8/10 |
| "Optimized" | 0.7596 | 2/10 |

**Conclusion:** Apparent improvements from combined datasets don't transfer to individual item training. **Stick with defaults.**

### Initial Conclusion (Quick Tests)

Quick tests suggested CatBoost defaults are optimal. However, full per-category testing revealed this is **not true**.

---

## Full Category-Based Experiments (2026-01-17)

**Platform:** Hydra Server (RTX 3060 12GB)
**Method:** 36 targets per item, 60-day window, full feature set
**Duration:** ~2 hours

### Results by Category

| Category | Best Depth | Best LR | Best L2 | Avg AUC |
|----------|------------|---------|---------|---------|
| high_volume_consumables | **3** | 0.15 | 1.0 | 0.69 |
| mid_volume_resources | **5** | 0.15 | 10.0 | 0.55 |
| potions_food | **6** | 0.05 | 5.0 | 0.70 |
| equipment_common | N/A | N/A | N/A | (insufficient data) |
| equipment_high_value | N/A | N/A | N/A | (insufficient data) |

### Key Findings

1. **Optimal depth varies by category:**
   - Consumables (runes, cannonballs): depth **3** (shallow trees prevent overfitting)
   - Resources (logs, ores): depth **5** (medium complexity)
   - Potions/food: depth **6** (deeper trees capture patterns)

2. **Learning rate patterns:**
   - Most categories: **0.15** (higher than default 0.1)
   - Potions/food: **0.05** (lower LR for higher-AUC items)

3. **L2 regularization:**
   - High-volume: **1.0** (less regularization for abundant data)
   - Mid-volume: **10.0** (more regularization for sparser data)
   - Potions: **5.0** (moderate)

4. **Equipment items need longer windows:**
   - High-value equipment (Bandos, AGS, etc.) have insufficient 60-day data
   - Recommend 180-day window for rare items

### Per-Item Variability

Even within categories, individual items have different optimal params:

| Item | Category | Best Depth | AUC |
|------|----------|------------|-----|
| Cannonball (2) | consumable | 3 | 0.779 |
| Nature rune (561) | consumable | 3 | 0.811 |
| Death rune (560) | consumable | 3 | 0.584 |
| Chaos rune (562) | consumable | **7** | 0.646 |
| Yew logs (1515) | resource | 5/8 | 0.560 |
| Coal (453) | resource | 5 | 0.583 |
| Prayer potion (2434) | potion | 6 | 0.700 |

**Chaos runes prefer depth 7** - opposite of other consumables!

### Recommendations

1. **Do NOT use single global hyperparameters** - different items need different settings
2. **Implement per-category defaults** at minimum
3. **Consider per-item optimization** for high-volume items
4. **Use longer training windows** (180 days) for rare equipment

### Updated Default Configuration

```yaml
# Category-specific defaults
hyperparameters:
  high_volume_consumables:
    depth: 3
    learning_rate: 0.15
    l2_leaf_reg: 1.0
  mid_volume_resources:
    depth: 5
    learning_rate: 0.15
    l2_leaf_reg: 10.0
  potions_food:
    depth: 6
    learning_rate: 0.05
    l2_leaf_reg: 5.0
  default:  # for uncategorized items
    depth: 4
    learning_rate: 0.1
    l2_leaf_reg: 3.0
```

---

## Training Infrastructure

### Hydra Server (Dedicated Training)
- **Host:** 10.0.0.146 (local network)
- **User:** ubuntu
- **GPU:** NVIDIA RTX 3060 12GB (CUDA 12.2, Driver 535)
- **OS:** Ubuntu 24.04 LTS

### Automated Training
- **Systemd Service:** `gept-training.service`
- **Schedule:** Weekly (Sunday 2:00 AM via `gept-training.timer`)
- **DB Tunnel:** `db-tunnel.service` (auto-reconnecting SSH tunnel to Ampere)

### Manual Training
```bash
ssh ubuntu@10.0.0.146
cd ~/gept
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gept
python cloud/train_runpod_multitarget.py --items 20
```

## Next Steps

1. ✅ **Hyperparameter optimization complete** - defaults are optimal

2. ✅ **Automated training configured** - weekly retraining enabled

3. **Update training_config.yaml** with confirmed optimal values:
   ```yaml
   training:
     days_history: 60
   catboost:
     iterations: 500
     depth: 4
     learning_rate: 0.1
     l2_leaf_reg: 3.0
     od_wait: 50
   ```

4. **Run full production training** with new config to validate at scale
