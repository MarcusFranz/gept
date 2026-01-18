# Tiered Training System

## Overview

Items are assigned to training tiers based on model quality (AUC). Higher-quality models train more frequently; low-quality models only train during monthly discovery runs.

## Tier Definitions

| Tier | AUC Range | Training Frequency | Rationale |
|------|-----------|-------------------|-----------|
| **A** | >= 0.58 | Daily | High-value predictable items |
| **B** | 0.54 - 0.58 | Every 3 days | Moderate signal |
| **C** | 0.52 - 0.54 | Weekly | Marginal but viable |
| **D** | < 0.52 | Monthly only | Check for improvement |

## Training Schedule

### Monthly Cycle

```
Day 1 (1st of month): FULL DISCOVERY RUN
├─ Train ALL items with sufficient data
├─ Reassign tiers based on fresh AUC
└─ Cost: ~$4.00 (with optimizations)

Days 2-30: TIERED TRAINING
├─ Tier A: Daily
├─ Tier B: Days 4, 7, 10, 13, 16, 19, 22, 25, 28
├─ Tier C: Days 8, 15, 22, 29
└─ Tier D: Skip (covered by monthly discovery)
```

### Estimated Monthly Cost

| Component | Items | Runs/Month | Cost |
|-----------|-------|------------|------|
| Discovery (Day 1) | 2,380 | 1 | $4.00 |
| Tier A (daily) | ~360 | 29 | $17.50 |
| Tier B (every 3 days) | ~595 | 10 | $9.70 |
| Tier C (weekly) | ~595 | 4 | $3.90 |
| **Total** | | | **~$35/month** |

## Tier State File

Stored in GCS at `gs://gept-models/tier_state.json`:

```json
{
  "last_updated": "2026-01-10T16:00:00Z",
  "last_discovery": "2026-01-01",
  "thresholds": {
    "A": 0.58,
    "B": 0.54,
    "C": 0.52
  },
  "items": {
    "565": {
      "name": "Blood rune",
      "tier": "A",
      "auc": 0.64,
      "last_trained": "2026-01-10",
      "consecutive_below": 0
    },
    "2": {
      "name": "Cannonball",
      "tier": "B",
      "auc": 0.56,
      "last_trained": "2026-01-09",
      "consecutive_below": 0
    }
  },
  "tier_counts": {
    "A": 360,
    "B": 595,
    "C": 595,
    "D": 830
  }
}
```

## Item Selection Logic

```python
def get_items_to_train_today(tier_state: dict, today: date) -> list:
    """Select items for today's training based on tier and schedule."""
    items_to_train = []

    # Check if this is a discovery day (1st of month)
    is_discovery_day = today.day == 1

    for item_id, item in tier_state['items'].items():
        tier = item['tier']
        last_trained = parse_date(item['last_trained'])
        days_since = (today - last_trained).days

        if is_discovery_day:
            # Discovery: train everything
            items_to_train.append(item_id)
        elif tier == 'A':
            items_to_train.append(item_id)  # Daily
        elif tier == 'B' and days_since >= 3:
            items_to_train.append(item_id)  # Every 3 days
        elif tier == 'C' and days_since >= 7:
            items_to_train.append(item_id)  # Weekly
        # Tier D: skip (only on discovery)

    return items_to_train
```

## Tier Transition Logic

Items can move between tiers based on AUC changes. Hysteresis prevents flip-flopping:

- **Promotion**: Move up immediately when AUC exceeds threshold
- **Demotion**: Require 2 consecutive runs below threshold

```python
THRESHOLDS = {'A': 0.58, 'B': 0.54, 'C': 0.52}
HYSTERESIS_RUNS = 2

def update_tier(item: dict, new_auc: float) -> str:
    """Update item tier based on new AUC."""
    old_tier = item['tier']

    # Determine target tier
    if new_auc >= THRESHOLDS['A']:
        target_tier = 'A'
    elif new_auc >= THRESHOLDS['B']:
        target_tier = 'B'
    elif new_auc >= THRESHOLDS['C']:
        target_tier = 'C'
    else:
        target_tier = 'D'

    # Promotions happen immediately
    if target_tier < old_tier:  # A < B < C < D in tier ordering
        return target_tier

    # Demotions require consecutive confirmations
    if target_tier > old_tier:
        consecutive = item.get('consecutive_below', 0) + 1
        if consecutive >= HYSTERESIS_RUNS:
            item['consecutive_below'] = 0
            return target_tier
        else:
            item['consecutive_below'] = consecutive
            return old_tier  # Stay in current tier

    # Same tier
    item['consecutive_below'] = 0
    return old_tier
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TIER_MODE` | Enable tiered training | `false` |
| `FORCE_DISCOVERY` | Force full discovery run | `false` |
| `RUN_ID` | Training run identifier | timestamp |
| `GCS_BUCKET` | GCS bucket name | `gept-models` |

## Usage

### Full Discovery Run
```bash
gcloud run jobs execute gept-daily-train \
  --set-env-vars "TIER_MODE=false,GCS_BUCKET=gept-models"
```

### Tiered Training Run
```bash
gcloud run jobs execute gept-daily-train \
  --set-env-vars "TIER_MODE=true,GCS_BUCKET=gept-models"
```

### Force Discovery (Mid-Month)
```bash
gcloud run jobs execute gept-daily-train \
  --set-env-vars "TIER_MODE=true,FORCE_DISCOVERY=true,GCS_BUCKET=gept-models"
```

## Design Rationale

### Why AUC-Based Tiers?

The training system only determines **model quality** (can we predict fills?). Capital allocation and GE limits are handled by the recommendation engine.

### Why Monthly Discovery?

Items can become predictable over time due to:
- New content releases changing trading patterns
- Meta shifts affecting item popularity
- Seasonal events creating predictable cycles

Monthly discovery ensures no item is permanently stuck in Tier D.

### Why Hysteresis?

Market volatility can cause AUC fluctuations. Requiring 2 consecutive demotions prevents wasting compute on items that temporarily dip below threshold.
