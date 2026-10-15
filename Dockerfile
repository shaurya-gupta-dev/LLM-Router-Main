# syntax=docker/dockerfile:1
# ==============================================================================
# LLM Router - Production Container Image
# Multi-Provider, Fault-Tolerant AI Routing Gateway
# ==============================================================================

FROM python:3.11-slim

LABEL org.opencontainers.image.title="LLM Router" \
      org.opencontainers.image.description="High-throughput, fault-tolerant LLM routing gateway aggregating 39+ AI providers into unified Anthropic and OpenAI endpoints." \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.authors="Shaurya Gupta"

# Optimize Python execution in container environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ROUTER_HOST=0.0.0.0 \
    ROUTER_PORT=8080

# Install essential runtime tools (curl for healthchecks, ca-certificates for TLS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create dedicated non-root application user
RUN useradd -u 1000 -m -s /bin/bash router && \
    chown -R router:router /app

# Copy dependency definition and install to system Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY --chown=router:router app/ ./app/
COPY --chown=router:router config/ ./config/
COPY --chown=router:router run.py .
COPY --chown=router:router .env.example .

# Expose standard gateway port
EXPOSE 8080

# Configure container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with least-privilege non-root user
USER router

# Launch gateway server
ENTRYPOINT ["python", "run.py"]
