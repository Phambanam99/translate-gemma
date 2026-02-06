@echo off
REM Script to install PyTorch CPU-only version on Windows
echo ========================================
echo Installing PyTorch (CPU only)
echo ========================================
echo.

REM Activate virtual environment
cd /d %~dp0
if exist .my-env\Scripts\activate.bat (
    call .my-env\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv .my-env
    call .my-env\Scripts\activate.bat
)

echo Installing PyTorch (CPU only)...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

echo.
echo ========================================
echo Testing installation...
echo ========================================
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo Done!
pause
