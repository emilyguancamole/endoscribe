# Multi-stage build for EndoScribe on Fly.io
# CPU-only (no WhisperX/GPU)
# Note: saved old Dockerfile with GPU support in `dockerfile-gpu`
# Multi-stage build for EndoScribe on Fly.io
# Uses rocker base + cached R packages layer for faster builds

### Frontend build
FROM node:20-slim AS frontend-builder
WORKDIR /build/frontend
COPY web_app/frontend/package.json web_app/frontend/package-lock.json* ./
RUN npm ci --silent
COPY web_app/frontend ./
RUN npm run build

### R packages layer - cached separately for faster rebuilds
FROM rocker/r-ver:4.3 AS r-packages
# Use RSPM for pre-compiled binaries (much faster than CRAN)
RUN R -e "options(repos = c(RSPM = 'https://packagemanager.posit.co/cran/__linux__/jammy/latest', CRAN = 'https://cloud.r-project.org')); install.packages(c('dplyr', 'caret', 'e1071', 'survival', 'class', 'lava', 'prodlim', 'ipred', 'recipes', 'FNN', 'gbm', 'lime'), Ncpus = 4)"

### Main application image
FROM rocker/r-ver:4.3 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
# Added: libpcre2-dev, liblzma-dev, libbz2-dev (needed for rpy2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    libicu-dev \
    gcc \
    g++ \
    gfortran \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy pre-installed R packages from r-packages stage
COPY --from=r-packages /usr/local/lib/R/site-library /usr/local/lib/R/site-library

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies including rpy2
RUN uv pip install --system --no-cache -r requirements.txt && \
    uv pip install --system --no-cache rpy2==3.5.11

# Download spacy model if needed
RUN python -m spacy download en_core_web_sm || echo "Spacy not in requirements, skipping"

# Clean up caches and temporary files
RUN rm -rf /root/.cache/pip \
    && rm -rf /root/.cache/uv \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf /usr/local/lib/R/site-library/*/help \
    && rm -rf /usr/local/lib/R/site-library/*/html \
    && rm -rf /usr/local/lib/R/site-library/*/doc \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local -name '*.pyc' -delete 2>/dev/null || true

# Create volume mount points for persistent storage
RUN mkdir -p /data/uploads /data/results /data/cache

# Set environment variables
ENV HF_HOME=/data/cache/huggingface
ENV TRANSFORMERS_CACHE=/data/cache/transformers
ENV PYTHONPATH=/app
ENV R_HOME=/usr/local/lib/R
ENV R_LIBS_USER=/usr/local/lib/R/site-library

# Copy application code
COPY . .

# Copy built frontend assets from the frontend-builder stage
COPY --from=frontend-builder /build/frontend/../static/dist web_app/static/dist

# Verify R and rpy2 installation
RUN python -c "import rpy2; print('rpy2 installed successfully')" && \
    python -c "from rpy2 import robjects; print('R integration working:', robjects.r('1+1')[0])"

# Expose port
EXPOSE 8000

# Run the web server
CMD ["python", "web_app/server.py"]