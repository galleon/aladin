# Stage 1: Builder stage - install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY backend/pyproject.toml ./

# Install Python dependencies to a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    sqlalchemy>=2.0.23 \
    psycopg2-binary>=2.9.9 \
    alembic>=1.13.1 \
    pydantic>=2.5.3 \
    pydantic-settings>=2.1.0 \
    pyyaml>=6.0.1 \
    structlog>=24.1.0 \
    prometheus-client>=0.19.0 \
    kubernetes==34.1.0 \
    celery>=5.3.4 \
    redis>=5.0.1 \
    websockets>=12.0

# Stage 2: Runtime stage - minimal final image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    postgresql-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Make sure we use venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY backend/app ./app
COPY backend/templates ./templates
COPY helm-charts ./helm-charts

# Install kubectl, clusterctl, and helm (runtime tools)
RUN curl -L https://dl.k8s.io/release/v1.30.0/bin/linux/amd64/kubectl -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl \
    && curl -L https://github.com/kubernetes-sigs/cluster-api/releases/download/v1.7.1/clusterctl-linux-amd64 -o /usr/local/bin/clusterctl \
    && chmod +x /usr/local/bin/clusterctl \
    && curl -L https://get.helm.sh/helm-v3.15.3-linux-amd64.tar.gz -o /tmp/helm.tar.gz \
    && tar -zxvf /tmp/helm.tar.gz -C /tmp \
    && mv /tmp/linux-amd64/helm /usr/local/bin/helm \
    && chmod +x /usr/local/bin/helm \
    && rm -rf /tmp/helm.tar.gz /tmp/linux-amd64

# Expose port
EXPOSE 3000

# Copy entrypoint script
COPY backend/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Start the application
CMD ["./entrypoint.sh"]
