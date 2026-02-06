@echo off
REM Setup script specifically for NVIDIA A4000 16GB with CUDA 12.7
echo ========================================
echo GPU Setup for NVIDIA A4000
echo CUDA 12.7 detected
echo ========================================
echo.

REM Check CUDA
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: CUDA toolkit not found in PATH
    echo Please ensure CUDA 12.7 is installed
    echo.
)

REM Check GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvidia-smi not found!
    echo Please install NVIDIA drivers
    pause
    exit /b 1
)

echo GPU Information:
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo.
echo [1/3] Installing PyTorch with CUDA 12.4 (compatible with CUDA 12.7)...
if exist ..\packages (
    echo Installing from local cache...
    pip install --no-index --find-links=..\packages torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
) else (
    echo Installing from PyTorch repository (requires internet)...
    pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
)

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PyTorch with CUDA
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB' if torch.cuda.is_available() else '')"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to verify CUDA
    pause
    exit /b 1
)

echo.
echo [3/3] Configuring for 16GB VRAM...
echo With 16GB VRAM, model will use float16 (no quantization needed)
echo This provides best performance and quality.

echo.
echo ========================================
echo GPU Setup completed!
echo ========================================
echo.
echo Your system:
echo - GPU: NVIDIA A4000
echo - VRAM: 16GB
echo - CUDA: 12.7
echo.
echo Model will run in float16 mode (optimal for 16GB VRAM)
echo.
pause
