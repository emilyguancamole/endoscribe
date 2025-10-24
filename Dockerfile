# Multi-stage build for EndoScribe on Fly.io with GPU support
# Base: NVIDIA CUDA 12.2 on Ubuntu 22.04 for WhisperX compatibility

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

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

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: torch will install CUDA-enabled version automatically
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Create volume mount points for persistent storage
RUN mkdir -p /data/models /data/uploads /data/results /data/cache

# Set environment variables for model caching
ENV HF_HOME=/data/cache/huggingface
ENV TRANSFORMERS_CACHE=/data/cache/transformers
ENV TORCH_HOME=/data/cache/torch
ENV WHISPERX_MODELS=/data/models/whisperx

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the web server
CMD ["python", "web_app/server.py"]
