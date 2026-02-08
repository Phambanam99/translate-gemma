#!/bin/bash

# ============================================================================
# TranslateGemma Docker Stop Script
# Usage: bash docker-stop.sh
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           Stopping TranslateGemma Docker Services              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Stop services
echo "[1/3] Stopping containers..."
if docker compose down; then
    echo "✓ Containers stopped"
else
    echo "❌ Failed to stop containers"
    exit 1
fi
echo ""

# Show status
echo "[2/3] Current status:"
docker compose ps
echo ""

# Show remaining volumes
echo "[3/3] Persistent volumes (not deleted):"
echo "  Use 'docker compose down -v' to remove all data"
docker volume ls | grep translate || echo "  No volumes found"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           ✓ Services Stopped Successfully                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
