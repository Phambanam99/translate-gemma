# Single-stage: CUDA runtime + Python — avoids missing .so errors from multi-stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# ── System packages (one layer, sorted for cache) ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    python3 python3-dev python3-pip python3-venv \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv ────────────────────────────────────────────────────────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# ── PyTorch (heaviest layer – cached unless base image changes) ────────
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Python deps (changes less often than app code) ─────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Application code (changes most often – last layer) ─────────────────
COPY . .
RUN mkdir -p uploads outputs model_cache

# ── Runtime defaults (overridden by docker-compose env) ────────────────
# When docker-compose maps a single GPU via device_ids, it appears as device 0
ENV GPU_DEVICE_ID=0 \
    CUDA_VISIBLE_DEVICES=0 \
    TRANSFORMERS_CACHE=/app/model_cache \
    HF_HOME=/app/model_cache

EXPOSE 8028

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -sf http://localhost:8028/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8028", \
    "--timeout-keep-alive", "120", "--limit-concurrency", "50"]
