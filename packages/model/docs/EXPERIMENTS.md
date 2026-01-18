# GePT Model Experiments

This document describes how to run A/B experiments to test model training configurations.

## Training Window Experiment

**Hypothesis:** For 48-hour prediction horizons, recent data (1 month) may outperform 6 months of historical data due to:
- Game meta changes (updates, nerfs, new content)
- Seasonal player behavior shifts
- Evolution of market manipulation patterns

### Prerequisites

1. Run the migration to create experiment tracking tables:
   ```bash
   psql -d osrs_data -f scripts/migrations/012_experiments.sql
   ```

2. Ensure database access:
   ```bash
   export DB_PASS="your_password"
   ```

### Running the Experiment

**Quick start (50 items, local CPU):**
```bash
python scripts/run_training_window_experiment.py --items 50 --local
```

**Full experiment (50 items, remote GPU):**
```bash
python scripts/run_training_window_experiment.py --items 50
```

**Dry run (see what would happen):**
```bash
python scripts/run_training_window_experiment.py --items 50 --dry-run
```

**Custom window comparison:**
```bash
# Compare 3 months vs 2 weeks
python scripts/run_training_window_experiment.py \
    --control-months 3 \
    --treatment-months 0.5 \
    --items 100
```

### Analyzing Results

After the experiment completes:
```bash
python scripts/run_training_window_experiment.py --analyze <experiment_id>
```

Example output:
```
============================================================
EXPERIMENT RESULTS
============================================================
Control (6mo):   mean_auc = 0.5842
Treatment (1mo): mean_auc = 0.5923
Difference:      +0.0081 (+1.39%)
Winner:          TREATMENT
============================================================
```

### Database Tables

- `experiments` - Experiment metadata and status
- `experiment_variants` - Per-variant configuration and results

Query experiment history:
```sql
SELECT experiment_id, name, status,
       results->>'comparison'->>'winner' as winner,
       created_at
FROM experiments
ORDER BY created_at DESC;
```

### Interpreting Results

| AUC Difference | Interpretation |
|----------------|----------------|
| < 0.5% | No meaningful difference |
| 0.5% - 1% | Minor improvement, consider other factors |
| 1% - 2% | Meaningful improvement, worth adopting |
| > 2% | Strong improvement, adopt with confidence |

### Next Steps Based on Results

**If treatment (shorter window) wins:**
1. Update `config/training_config.yaml`: `months_history: 1`
2. Consider even shorter windows (2 weeks, 1 week)
3. Increase retraining frequency to keep models fresh

**If control (longer window) wins:**
1. Keep current 6-month window
2. Consider sample weighting to favor recent data
3. Explore hybrid approaches (6mo base + recent fine-tuning)

### Running Multiple Experiments

You can run experiments with different parameters:

```bash
# Window length sweep
for months in 1 2 3 6; do
    python scripts/run_training_window_experiment.py \
        --treatment-months $months \
        --control-months 6 \
        --items 30
done
```

Then compare all experiments:
```sql
SELECT
    e.experiment_id,
    v.variant_name,
    v.config->>'months_history' as months,
    v.mean_auc
FROM experiments e
JOIN experiment_variants v ON e.experiment_id = v.experiment_id
WHERE e.name LIKE 'Training Window%'
ORDER BY v.mean_auc DESC;
```
