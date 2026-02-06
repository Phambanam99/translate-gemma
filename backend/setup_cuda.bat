@echo off
REM Script to install PyTorch with CUDA support on Windows
echo ========================================
echo Installing PyTorch with CUDA 12.4
echo ========================================
echo.

REM Check if CUDA is available
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: CUDA toolkit not found in PATH
    echo Please install CUDA toolkit first
    echo.
)

REM Activate virtual environment
cd /d %~dp0
if exist .my-env\Scripts\activate.bat (
    call .my-env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv .my-env
    call .my-env\Scripts\activate.bat
)

echo Installing PyTorch with CUDA 12.4...
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo.
echo ========================================
echo Testing CUDA installation...
echo ========================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Done!
pause
