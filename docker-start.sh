#!/bin/bash
# ============================================================================
# TranslateGemma – one-command Docker deploy
# Usage: bash docker-start.sh [gpu-id]   (default: 2)
# ============================================================================
set -euo pipefail

GPU_ID=${1:-2}
PORT=8028

echo "╔═══════════════════════════════════════════════════╗"
echo "║  TranslateGemma  ·  GPU $GPU_ID  ·  port $PORT        ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────
echo "[1/4] Checking Docker…"
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "❌ Docker Compose not found"; exit 1; }
echo "  ✓ Docker $(docker --version | grep -oP '\d+\.\d+\.\d+')"

echo "[2/4] Checking NVIDIA runtime…"
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi >/dev/null 2>&1; then
    echo "  ⚠  nvidia-docker test failed – GPU may not work inside container"
else
    echo "  ✓ NVIDIA runtime OK"
fi

# ── Build ─────────────────────────────────────────────────────────────
echo "[3/4] Building images (cached layers reused)…"
docker compose build

# ── Launch ────────────────────────────────────────────────────────────
echo "[4/4] Starting services (GPU $GPU_ID → container device 0)…"
docker compose up -d

echo ""
echo "══════════════════════════════════════════"
echo "  ✓  http://localhost:$PORT"
echo "  ✓  API docs  http://localhost:$PORT/api/docs"
echo "  GPU $GPU_ID  (A100 80 GB)"
echo "══════════════════════════════════════════"
echo ""
echo "  logs:   docker compose logs -f backend"
echo "  stop:   docker compose down"
echo "  gpu:    docker exec translate-gemma-backend nvidia-smi"
