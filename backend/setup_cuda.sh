#!/bin/bash
# Script to install PyTorch with CUDA support on Linux/Mac

echo "========================================"
echo "Installing PyTorch with CUDA 12.4"
echo "========================================"
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: CUDA toolkit not found in PATH"
    echo "Please install CUDA toolkit first"
    echo ""
fi

# Activate virtual environment
cd "$(dirname "$0")"
if [ -d ".my-env" ]; then
    source .my-env/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .my-env
    source .my-env/bin/activate
fi

echo "Installing PyTorch with CUDA 12.4..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "========================================"
echo "Testing CUDA installation..."
echo "========================================"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "Done!"
