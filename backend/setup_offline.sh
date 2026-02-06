#!/bin/bash
# Setup script for offline installation (Linux/Mac)

echo "========================================"
echo "CSV Translator - Offline Setup"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found!"
    echo "Please install Python 3.11+ first"
    exit 1
fi

echo "[1/5] Creating virtual environment..."
if [ -d ".my-env" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf .my-env
fi
python3 -m venv .my-env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi
echo "✓ Virtual environment created"

echo ""
echo "[2/5] Activating virtual environment..."
source .my-env/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "✓ Virtual environment activated"

echo ""
echo "[3/5] Installing Python packages from local cache..."
if [ -d "../packages" ]; then
    echo "Installing packages (this may take a while)..."
    pip install --no-index --find-links=../packages -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "WARNING: Some packages failed to install from cache"
        echo "Trying to install from requirements-full.txt..."
        if [ -f "../requirements-full.txt" ]; then
            pip install --no-index --find-links=../packages -r ../requirements-full.txt
        fi
    fi
    echo "✓ Packages installed"
else
    echo "WARNING: packages/ directory not found"
    echo "Attempting to install from requirements.txt (requires internet)..."
    pip install -r requirements.txt
fi

echo ""
echo "[4/5] Setting up model cache..."
if [ -d "../model_cache/models--google--translategemma-4b-it" ]; then
    echo "Copying model cache..."
    CACHE_DIR="$HOME/.cache/huggingface/hub"
    mkdir -p "$CACHE_DIR"
    cp -r "../model_cache/models--google--translategemma-4b-it" "$CACHE_DIR/"
    echo "✓ Model cache installed"
else
    echo "WARNING: Model cache not found in package"
    echo "You will need to download the model when you have internet"
    echo "Run: python download_model.py"
fi

echo ""
echo "[5/5] Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Offline mode
HF_HUB_OFFLINE=true
TRANSFORMERS_OFFLINE=true
EOF
    echo "✓ .env file created"
else
    echo ".env file already exists"
fi

echo ""
echo "========================================"
echo "Setup completed!"
echo "========================================"
echo ""
echo "To start the server:"
echo "  1. cd backend"
echo "  2. source .my-env/bin/activate"
echo "  3. python main.py"
echo ""
echo "Or run: ./run_offline.sh"
echo ""
