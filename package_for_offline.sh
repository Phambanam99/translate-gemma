#!/bin/bash
# Script to package everything for offline installation (Linux/Mac)

PACKAGE_NAME="csv-translator-offline"
PACKAGE_DIR="${PACKAGE_NAME}-$(date +%Y%m%d)"

echo "========================================"
echo "Packaging for OFFLINE installation"
echo "========================================"
echo ""

echo "Creating package directory: $PACKAGE_DIR"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

echo ""
echo "[1/6] Copying source code..."
cp -r backend "$PACKAGE_DIR/"
cp -r frontend "$PACKAGE_DIR/"
echo "✓ Code copied"

echo ""
echo "[2/6] Exporting Python packages..."
cd backend
if [ -d ".my-env" ]; then
    source .my-env/bin/activate
    echo "Exporting package list..."
    pip freeze > "../$PACKAGE_DIR/requirements-full.txt"
    echo "Downloading all packages (this may take a while)..."
    pip download -r requirements.txt -d "../$PACKAGE_DIR/packages" --no-deps
    pip download -d "../$PACKAGE_DIR/packages" torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
    echo "✓ Packages downloaded"
else
    echo "WARNING: Virtual environment not found. Skipping package export."
fi
cd ..

echo ""
echo "[3/6] Copying model cache..."
CACHE_DIR="$HOME/.cache/huggingface/hub"
if [ -d "$CACHE_DIR/models--google--translategemma-4b-it" ]; then
    echo "Copying model cache (this may take a while, ~10GB)..."
    mkdir -p "$PACKAGE_DIR/model_cache"
    cp -r "$CACHE_DIR/models--google--translategemma-4b-it" "$PACKAGE_DIR/model_cache/"
    echo "✓ Model cache copied"
else
    echo "WARNING: Model cache not found at $CACHE_DIR"
    echo "Please run: python backend/download_model.py first"
    mkdir -p "$PACKAGE_DIR/model_cache"
    echo "Model cache not found. Please download model first." > "$PACKAGE_DIR/model_cache/README.txt"
fi

echo ""
echo "[4/6] Creating setup scripts..."
cp backend/setup_offline.sh "$PACKAGE_DIR/setup.sh"
cp backend/INSTALL_OFFLINE.md "$PACKAGE_DIR/INSTALL.md"
chmod +x "$PACKAGE_DIR/setup.sh"
echo "✓ Setup scripts created"

echo ""
echo "[5/6] Creating package info..."
cat > "$PACKAGE_DIR/PACKAGE_INFO.txt" << EOF
Package created: $(date)
Source machine: $(hostname)
Python version: $(python --version)

Contents:
- backend/ : Source code
- frontend/ : Frontend code  
- packages/ : Python packages (wheels)
- model_cache/ : Hugging Face model cache
- requirements-full.txt : Full package list
- setup.sh : Installation script
- INSTALL.md : Installation instructions
EOF
cat "$PACKAGE_DIR/PACKAGE_INFO.txt"

echo ""
echo "[6/6] Creating README..."
cat > "$PACKAGE_DIR/README.txt" << EOF
========================================
CSV Translator - Offline Package
========================================

This package contains everything needed to run
CSV Translator on a machine WITHOUT internet.

QUICK START:
1. Extract this folder to your target machine
2. Run: chmod +x setup.sh && ./setup.sh
3. Run: cd backend && source .my-env/bin/activate && python main.py

See INSTALL.md for detailed instructions.
EOF

echo ""
echo "========================================"
echo "Package created: $PACKAGE_DIR"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Compress $PACKAGE_DIR to .tar.gz or .zip"
echo "2. Copy to target machine"
echo "3. Extract and run ./setup.sh"
echo ""
echo "Package size:"
du -sh "$PACKAGE_DIR"
