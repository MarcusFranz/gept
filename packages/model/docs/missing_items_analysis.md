# Missing Items Analysis

**Generated:** January 8, 2026
**Items Analyzed:** 25 items without trained models

---

## Executive Summary

The 25 items without models are NOT untradeable - they simply need **lower offsets** (1% or 1.5% instead of 2%/2.5%). Most of these are high-value items with excellent fill rates at 1% offset:

| Item | 2% @ 24h Fill | 1% @ 24h Fill | Verdict |
|------|---------------|---------------|---------|
| Rune platebody | 21.1% | **61.2%** | Add 1% models |
| Rune platelegs | 17.0% | **56.3%** | Add 1% models |
| Runite ore | 16.5% | **61.7%** | Add 1% models |
| Berserker ring | 16.6% | **44.9%** | Add 1% models |
| Abyssal whip | 6.9% | **38.4%** | Add 1% models |
| Amulet of fury | 12.5% | **41.2%** | Add 1% models |
| Zulrah's scales | 12.3% | **42.5%** | Add 1% models |
| Ranarr weed | 4.8% | **36.7%** | Add 1% models |
| Runite bar | 0.2% | **1.5%** | Maybe skip |
| Feather | 0.4% | 0.4% | Skip (price too low) |

**Recommendation:** Train 1% and 1.5% offset models for these items. Expected additional ~1,000 models (25 items × 24 hours × ~2 offsets).

---

## 1. Why These Items Have No Models

### Training Configuration
The training pipeline only used:
- **Offsets:** 2% and 2.5%
- **Hours:** 1-24

For these stable items, fill rates at 2%+ offsets are below the training threshold (need positive samples).

### Item Categories

**Category 1: High-Value End-Game Gear**
- Abyssal whip (1.5M GP avg)
- Amulet of fury (2.3M GP avg)
- Berserker ring (4.4M GP avg)
- Dragon platelegs/plateskirt

*Characteristics:* High value, active traders, tight spreads (~2%), low volatility at 2%+ offset

**Category 2: Alch-Locked Items**
- Rune platebody (38k GP avg)
- Rune platelegs (38k GP avg)
- Rune 2h sword

*Characteristics:* Price capped by high-alch value, small spread (0.3-0.4%), stable

**Category 3: High-Volume Commodities**
- Runite bar (12k GP avg)
- Runite ore (10k GP avg)
- Feather (2 GP avg)
- Ranarr weed (7k GP avg)

*Characteristics:* Very liquid, efficient pricing, small spreads

**Category 4: Specialized Items**
- Zulrah's scales (213 GP avg)
- Prayer potion(4)
- Skills necklace variants

---

## 2. Volatility Analysis by Item

### Abyssal Whip (ID: 4151)
```
Average Price: 1,466,787 GP
Average Spread: 1.86%

Range Analysis (% of time range exceeds threshold):
  1h range: 2%+ = 99.2%, 4%+ = 7.6%
  4h range: 2%+ = 99.9%, 4%+ = 28.6%
  24h range: 2%+ = 99.5%, 4%+ = 89.8%

Roundtrip Fill Rates:
  2.0% @ 1h: 0.15%  │  2.0% @ 24h: 6.86%
  1.5% @ 1h: 0.57%  │  1.5% @ 24h: 17.51%
  1.0% @ 1h: 2.54%  │  1.0% @ 24h: 38.44%  ← EXCELLENT
```

**Verdict:** Train 1% and 1.5% models. High fill rates at lower offsets.

---

### Berserker Ring (ID: 6737)
```
Average Price: 4,362,811 GP
Average Spread: 2.06%

Roundtrip Fill Rates:
  2.0% @ 1h: 0.30%  │  2.0% @ 24h: 16.58%
  1.5% @ 1h: 0.92%  │  1.5% @ 24h: 28.06%
  1.0% @ 1h: 3.30%  │  1.0% @ 24h: 44.94%  ← EXCELLENT
```

**Verdict:** Train 1% and 1.5% models. Even 2% @ 24h has good rates.

---

### Rune Platebody (ID: 1127)
```
Average Price: 38,501 GP
Average Spread: 0.39% (alch-locked)

Roundtrip Fill Rates:
  2.0% @ 1h: 0.11%  │  2.0% @ 24h: 21.08%
  1.5% @ 1h: 0.38%  │  1.5% @ 24h: 31.82%
  1.0% @ 1h: 2.07%  │  1.0% @ 24h: 61.22%  ← EXCELLENT
```

**Verdict:** Train 1% models. Very high fill rates.

---

### Zulrah's Scales (ID: 12934)
```
Average Price: 213 GP
Average Spread: 1.99%

Roundtrip Fill Rates:
  2.0% @ 1h: 0.28%  │  2.0% @ 24h: 12.27%
  1.5% @ 1h: 1.09%  │  1.5% @ 24h: 23.63%
  1.0% @ 1h: 4.80%  │  1.0% @ 24h: 42.52%  ← EXCELLENT
```

**Verdict:** Train 1% and 1.5% models.

---

### Runite Bar (ID: 2363)
```
Average Price: 12,238 GP
Average Spread: 0.17% (very tight!)

Roundtrip Fill Rates:
  2.0% @ 1h: 0.00%  │  2.0% @ 24h: 0.16%
  1.5% @ 4h: 0.12%  │  1.5% @ 24h: 0.44%
  1.0% @ 1h: 0.26%  │  1.0% @ 24h: 1.51%  ← LOW
```

**Verdict:** Maybe skip. Fill rates are low even at 1%.

---

### Feather (ID: 314)
```
Average Price: 2 GP
Average Spread: 34.9% (price too low for meaningful %)

Roundtrip Fill Rates: ~0.4% across all offsets
```

**Verdict:** Skip. Price is too low (2 GP) for percentage-based trading.

---

## 3. Economic Analysis

### Profit at Different Offsets

| Offset | Gross Profit | GE Tax | Net Profit |
|--------|-------------|--------|------------|
| 2.5% | 5.0% | 2.0% | **3.0%** |
| 2.0% | 4.0% | 2.0% | **2.0%** |
| 1.5% | 3.0% | 2.0% | **1.0%** |
| 1.0% | 2.0% | 2.0% | **0.0%** |

**⚠️ At 1% offset, profit = 0% (break-even)**

This means:
- 1% offset trades only make money from timing (buying low, selling high within the offset)
- Need to reconsider the profit calculation for 1% offset trades
- OR the tax model is wrong (tax is 1% per side, not 2% total)

### Corrected Tax Model

GE Tax is 1% on the **sell** side only:
- Buy at 1000 GP
- Sell at 1020 GP (2% markup)
- Tax: 10.2 GP (1% of 1020)
- Net profit: 1020 - 1000 - 10.2 = **9.8 GP** (0.98%)

For 1% offset:
- Buy at 1000 GP
- Sell at 1010 GP (1% markup)
- Tax: 10.1 GP
- Net profit: 1010 - 1000 - 10.1 = **-0.1 GP** (breakeven)

**Conclusion:** 1% offset is not profitable after tax. Use 1.5% minimum.

---

## 4. Recommendations

### Items to Train with 1.5% Offset

| Item | 1.5% @ 24h Fill | Net Profit | EV @ 24h |
|------|----------------|------------|----------|
| Abyssal whip | 17.5% | 1.0% | 0.175% |
| Berserker ring | 28.1% | 1.0% | 0.281% |
| Amulet of fury | 23.7% | 1.0% | 0.237% |
| Rune platebody | 31.8% | 1.0% | 0.318% |
| Rune platelegs | 31.8% | 1.0% | 0.318% |
| Runite ore | 39.5% | 1.0% | 0.395% |
| Zulrah's scales | 23.6% | 1.0% | 0.236% |
| Ranarr weed | 15.9% | 1.0% | 0.159% |

**Total: 8 items × 24 hours × 1 offset = 192 new models**

### Items to Skip

| Item | Reason |
|------|--------|
| Runite bar | Fill rates too low even at 1.5% |
| Feather | Price too low (2 GP) |
| Prayer potion(4) | Needs investigation |
| Bird nest (empty) | Needs investigation |
| Skills necklace variants | Needs investigation |
| Dragon crossbow | Needs investigation |
| Bracelet of ethereum | Needs investigation |

### Training Plan

```python
# Add to training config
additional_items = [4151, 6737, 6585, 1127, 1079, 451, 12934, 257]
additional_offsets = [0.015]  # 1.5% only (1% not profitable)
hours = list(range(1, 25))

# Expected: 8 items × 24 hours = 192 new models
# Training time: ~1 hour with Cloud Run
```

---

## 5. Implementation Steps

1. **Verify tax model** - Confirm GE tax is 1% on sell only
2. **Update training config** - Add 1.5% offset option
3. **Re-run training** - For the 8 recommended items
4. **Validate new models** - Run backtest on 1.5% offset models
5. **Deploy** - Add to prediction pipeline

---

## 6. Conclusion

The 25 "missing" items are not inherently untradeable - they just need lower offset models. By training with 1.5% offset (the minimum profitable level), we can add ~8 popular high-value items to the tradeable pool.

**Priority:** High for Berserker ring, Rune platebody, Runite ore (best fill rates)

---

*Analysis performed using price_data_5min, June 15, 2025 - January 6, 2026*
