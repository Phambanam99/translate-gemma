#!/bin/bash
# Script to run server in offline mode (Linux/Mac)
# Make sure you've downloaded the model first using: python download_model.py

echo "========================================"
echo "Starting server in OFFLINE mode"
echo "========================================"
echo ""

# Set offline mode environment variables
export HF_HUB_OFFLINE=true
export TRANSFORMERS_OFFLINE=true

# Activate virtual environment and run server
cd "$(dirname "$0")"
source .my-env/bin/activate
python main.py
