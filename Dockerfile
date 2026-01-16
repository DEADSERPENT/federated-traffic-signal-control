# ==============================================================================
# Federated Learning-based Adaptive Traffic Signal Control System
# Docker Container
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/results/comprehensive /app/results/logs

# Set permissions
RUN chmod +x /app/*.py 2>/dev/null || true

# Default command - run comprehensive experiment
CMD ["python", "run_comprehensive.py", "--quick"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import flwr; print('OK')" || exit 1

# Labels
LABEL maintainer="Traffic Signal Control Team" \
      version="1.0.0" \
      description="Federated Learning-based Adaptive Traffic Signal Control System"
