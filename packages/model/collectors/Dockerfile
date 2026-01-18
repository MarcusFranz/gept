# =============================================================================
# GePT Collectors - Multi-arch Dockerfile
# =============================================================================
# Single image for all collectors. Entry point selected via CMD override.
# Supports both AMD64 (Vast.ai) and ARM64 (Oracle Cloud Ampere).
# =============================================================================

FROM python:3.10-slim

# Build argument for collector selection
ARG COLLECTOR=collect_data_v2.py

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Rationale: Using pip directly (not poetry) for smaller image size in production
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy collector scripts
COPY *.py ./

# Create non-root user for security
RUN groupadd --gid 1000 gept \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash gept \
    && mkdir -p /data \
    && chown -R gept:gept /app /data

USER gept

# Data volume mount point
VOLUME /data

# Default metrics port (overridden per collector)
EXPOSE 9100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${METRICS_PORT:-9100}/metrics || exit 1

# Default entry point (override with docker-compose command)
CMD ["python", "collect_data_v2.py"]
