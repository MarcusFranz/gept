# GePT3 CUDA Training (Rented Server)

## 0) Production-grade full run (recommended)

For a full production-grade run (365d, 5m, full item universe intent), use:

```bash
cd /path/to/gept
source packages/inference/scripts/gept3/production_profile_full.env.example
packages/inference/scripts/gept3/run_gept3_production_pipeline.sh
```

Runbook:
- `packages/inference/scripts/gept3/PRODUCTION_RUNBOOK.md`

Pipeline stages:
- `STAGE=row` for uncapped row dataset build
- `STAGE=hybrid` for hybrid snapshot build
- `STAGE=train` for resumable CUDA training
- `STAGE=all` for end-to-end

## 1) Native CUDA run (no container)

1. Create venv and install deps:
```bash
cd /path/to/gept/packages/inference
python3 -m venv .venv312
source .venv312/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Blackwell-safe bootstrap (recommended):
```bash
cd /path/to/gept
packages/inference/scripts/gept3/setup_training_env_cuda.sh
```

2. Launch resumable training:
```bash
cd /path/to/gept
packages/inference/scripts/gept3/run_hybrid_cuda_scaleup.sh
```

3. Launch watchdog (optional):
```bash
cd /path/to/gept
packages/inference/scripts/gept3/start_cuda_watchdog.sh
```

## 2) Containerized CUDA run

1. Build + run with one command:
```bash
cd /path/to/gept
packages/inference/scripts/gept3/run_cuda_training_container.sh
```

2. Override run settings:
```bash
RUN_NAME=gept3_hybrid_cuda_full_365d \
DATASET=/workspace/gept/packages/inference/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz \
EPOCHS=16 BATCH_SIZE=2 GRAD_ACCUM_STEPS=6 \
packages/inference/scripts/gept3/run_cuda_training_container.sh
```

## 3) Publish image to Docker Hub

1. Build and push the training image:
```bash
cd /path/to/gept
DOCKERHUB_USERNAME=your_dockerhub_user \
TAG=v1 \
packages/inference/scripts/gept3/publish_dockerhub_train_image.sh
```

2. Optional explicit repo name:
```bash
IMAGE_REPO=your_dockerhub_user/gept3-train \
TAG=v1 \
packages/inference/scripts/gept3/publish_dockerhub_train_image.sh
```

Notes:
- Default repo is `${DOCKERHUB_USERNAME}/gept3-train`.
- By default this also pushes `:latest` (`PUSH_LATEST=1`).
- You can authenticate non-interactively with `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`.

## 4) Publish all-in-one image (includes dataset files)

If you want a single image with training code + bundled dataset(s), use:

```bash
cd /path/to/gept
DOCKERHUB_USERNAME=your_dockerhub_user \
TAG=allinone-v1 \
DATASET_PATHS=packages/inference/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz \
packages/inference/scripts/gept3/publish_dockerhub_train_image_allinone.sh
```

Options:
- `INCLUDE_ALL_DATA=1` to include the entire `packages/inference/data` folder.
- `IMAGE_REPO=your_user/gept3-train-allinone` to override repo name.
- `PUSH=0` to build locally without pushing.

## 5) Deploy straight from Docker Hub (GPU instance)

1. Pull and run:
```bash
IMAGE_REPO=your_dockerhub_user/gept3-train \
TAG=v1 \
HOST_BASE=/opt/gept/inference \
RUN_NAME=gept3_hybrid_cuda_full_365d \
packages/inference/scripts/gept3/deploy_dockerhub_train_container.sh
```

For all-in-one image (use bundled in-image dataset, no data mount):
```bash
IMAGE_REPO=your_dockerhub_user/gept3-train-allinone \
TAG=allinone-v1 \
MOUNT_DATA=0 \
HOST_BASE=/opt/gept/inference \
RUN_NAME=gept3_hybrid_cuda_full_365d \
packages/inference/scripts/gept3/deploy_dockerhub_train_container.sh
```

2. Watch training progress:
```bash
tail -f /opt/gept/inference/logs/gept3_hybrid_cuda_full_365d.log
```

3. Stop/restart container:
```bash
docker stop gept3-train
docker start gept3-train
```

## 6) Deploy via Docker Compose (optional)

1. Copy/edit compose template:
```bash
cd /path/to/gept/packages/inference/scripts/gept3
cp docker-compose.dockerhub-train.yml docker-compose.train.yml
```

2. Set environment (or `.env`) and run:
```bash
IMAGE=docker.io/your_dockerhub_user/gept3-train:v1 \
HOST_DATA_DIR=/opt/gept/inference/data \
HOST_MODELS_DIR=/opt/gept/inference/models \
HOST_LOGS_DIR=/opt/gept/inference/logs \
docker compose -f docker-compose.train.yml up -d
```

All-in-one compose template (no data volume mount):
```bash
cd /path/to/gept/packages/inference/scripts/gept3
IMAGE=docker.io/your_dockerhub_user/gept3-train-allinone:allinone-v1 \
HOST_MODELS_DIR=/opt/gept/inference/models \
HOST_LOGS_DIR=/opt/gept/inference/logs \
docker compose -f docker-compose.dockerhub-train-allinone.yml up -d
```

## Resume behavior

- Training checkpoints are saved to:
  - `packages/inference/models/gept3/<RUN_NAME>/checkpoints/latest.pt`
- Resume is enabled by default in launch scripts (`AUTO_RESUME=1`), so interrupted runs continue from the next epoch.
- Force resume from a specific checkpoint:
```bash
RESUME_FROM=/abs/path/to/latest.pt packages/inference/scripts/gept3/run_hybrid_cuda_scaleup.sh
```

## Notes

- The training image is CUDA-focused and uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.
- Host directories `packages/inference/data`, `packages/inference/models`, and `packages/inference/logs` are mounted into the container.
- Keep `RUN_NAME` stable to preserve checkpoint continuity.
