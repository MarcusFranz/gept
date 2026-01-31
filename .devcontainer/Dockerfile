FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 from deadsnakes PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        # Build deps for psycopg2
        gcc \
        libpq-dev \
        # PostgreSQL client for debugging
        postgresql-client \
        # SSH client for Ampere tunnel
        openssh-client \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install Node 20 via nodesource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
