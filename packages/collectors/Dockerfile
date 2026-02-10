FROM python:3.12-slim

# Install curl for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy collector scripts and shared modules
COPY *.py ./
COPY shared/ ./shared/

# Copy Prometheus configs (for volume mount reference)
COPY prometheus/ ./prometheus/

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

# Default entrypoint (overridden per service in docker-compose)
CMD ["python", "collect_5m_pg.py"]
