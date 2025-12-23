# Multi-stage build for EndoScribe on Fly.io with GPU support
# Base: NVIDIA CUDA 12.2 on Ubuntu 22.04 for WhisperX compatibility

### Frontend build: produce `web_app/static/dist`
FROM node:20 AS frontend-builder
WORKDIR /build/frontend
COPY web_app/frontend/package.json web_app/frontend/package-lock.json* ./
COPY web_app/frontend ./
RUN npm ci --silent
RUN npm run build

### Base image with CUDA for WhisperX
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA support FIRST (before requirements.txt)
# This ensures we get CUDA-enabled PyTorch, not CPU version
RUN uv pip install --system --no-cache \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies using uv (much faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Verify PyTorch installation (CUDA availability checked at runtime, not build time)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA compiled: {torch.version.cuda}')"

# Create volume mount points for persistent storage
RUN mkdir -p /data/models /data/uploads /data/results /data/cache

# Set environment variables for model caching
ENV HF_HOME=/data/cache/huggingface
ENV TRANSFORMERS_CACHE=/data/cache/transformers
ENV TORCH_HOME=/data/cache/torch
ENV WHISPERX_MODELS=/data/models/whisperx

# Copy application code
COPY . .

# Copy built frontend assets from the frontend-builder stage into the final image
# Vite's config outputs to `web_app/static/dist` (relative to frontend dir), so copy that folder.
COPY --from=frontend-builder /build/frontend/../static/dist web_app/static/dist

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
# HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    # CMD curl -f http://localhost:8000/health || exit 1

# Run the web server
CMD ["python", "web_app/server.py"]

