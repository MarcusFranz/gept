FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/models/patchtst/best_model.pt \
    MIN_VOLUME=2000

# Minimal runtime deps:
# - libgomp1: required by numpy/scipy/torch wheels for OpenMP
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

# Default: run PatchTST inference (one-shot) and exit.
CMD ["python", "src/pipeline/run_patchtst_inference.py", "--model-path", "/models/patchtst/best_model.pt", "--min-volume", "2000"]

