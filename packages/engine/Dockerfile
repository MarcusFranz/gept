FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/
COPY pyproject.toml ./

# Create logs directory
RUN mkdir -p logs

# Environment variables (defaults)
ENV DB_CONNECTION_STRING=""
ENV DISCORD_WEBHOOK_URL=""
ENV MODEL_REGISTRY_PATH="models/registry.json"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
