# GePT3 Production-Grade Training Runbook

## Goal
Prepare and execute a full production-grade GePT3 training run with:
1. 365-day lookback
2. 5-minute granularity (`sample_every=1`)
3. Full item universe from latest predictions
4. Resumable CUDA training with checkpoints

## Reality check on dataset scale
- Current earlier run used capped row sampling (`max_rows_per_item=12000`) and capped hybrid snapshots.
- True 365d x 5m x ~1k items is roughly 110M row-level timesteps before filtering.
- You can run uncapped row dataset generation now, but hybrid snapshot count still needs a cap that fits host RAM.

## Recommended server tiers
1. Minimum viable full run: 24GB+ VRAM, 128GB RAM, 500GB disk
2. Preferred: 48-96GB VRAM, 256GB RAM, 1TB disk
3. Comfort tier for broad experiments/history: 96GB VRAM, 512GB RAM, 2TB disk

## Environment bootstrap (CUDA + Python)
```bash
cd /path/to/gept
packages/inference/scripts/gept3/setup_training_env_cuda.sh
```

Notes:
1. Default setup uses `TORCH_PROFILE=nightly-cu128` for Blackwell-class GPUs (`sm_120`).
2. For older GPUs, you can use `TORCH_PROFILE=stable-cu124`.

## One-command pipeline
```bash
cd /path/to/gept
source packages/inference/scripts/gept3/production_profile_full.env.example
packages/inference/scripts/gept3/run_gept3_production_pipeline.sh
```

## Stage-by-stage execution
1. Row-only build:
```bash
STAGE=row packages/inference/scripts/gept3/run_gept3_production_pipeline.sh
```
2. Hybrid-only build:
```bash
STAGE=hybrid packages/inference/scripts/gept3/run_gept3_production_pipeline.sh
```
3. Train-only:
```bash
STAGE=train packages/inference/scripts/gept3/run_gept3_production_pipeline.sh
```

## Resume behavior
- Checkpoints:
  - `packages/inference/models/gept3/<RUN_NAME>/checkpoints/latest.pt`
- `AUTO_RESUME=1` resumes automatically for stable `RUN_NAME`.

## Sizing knob that matters most
- `MAX_SNAPSHOTS` is the main RAM lever for the hybrid dataset.
- Suggested starting points:
  1. 128GB RAM: `MAX_SNAPSHOTS=12000`
  2. 256GB RAM: `MAX_SNAPSHOTS=18000`
  3. 512GB RAM: `MAX_SNAPSHOTS=24000` or higher

## Validation checkpoints
1. Row dataset summary printed after row stage
2. Hybrid dataset summary printed after hybrid stage
3. Training log at:
   - `packages/inference/logs/<RUN_NAME>.log`
4. Final artifacts at:
   - `packages/inference/models/gept3/<RUN_NAME>/`

## Notes
- If training must run from Docker Hub image directly, use scripts in `README_CUDA_TRAINING.md`.
- For all-in-one Docker images, set `MOUNT_DATA=0` so bundled dataset files are used.
