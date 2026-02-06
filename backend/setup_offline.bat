@echo off
REM Setup script for offline installation
echo ========================================
echo CSV Translator - Offline Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.11+ first
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
if exist .my-env (
    echo Virtual environment already exists. Removing...
    rmdir /s /q .my-env
)
python -m venv .my-env
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created

echo.
echo [2/5] Activating virtual environment...
call .my-env\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

echo.
echo [3/5] Installing Python packages from local cache...
if exist ..\packages (
    echo Installing packages (this may take a while)...
    
    REM First install PyTorch with CUDA if available
    echo Checking for CUDA PyTorch packages...
    if exist ..\packages\torch-2.6.0+cu124*.whl (
        echo Installing PyTorch with CUDA 12.4 (compatible with CUDA 12.7)...
        pip install --no-index --find-links=..\packages torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
        if %ERRORLEVEL% EQU 0 (
            echo ✓ PyTorch CUDA installed
        ) else (
            echo WARNING: CUDA PyTorch installation failed, trying CPU version...
            pip install --no-index --find-links=..\packages torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
        )
    ) else (
        echo Installing PyTorch CPU version...
        pip install --no-index --find-links=..\packages torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
    )
    
    REM Install other packages
    echo Installing other packages...
    pip install --no-index --find-links=..\packages -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Some packages failed to install from cache
        echo Trying to install from requirements-full.txt...
        if exist ..\requirements-full.txt (
            pip install --no-index --find-links=..\packages -r ..\requirements-full.txt
        )
    )
    echo ✓ Packages installed
) else (
    echo WARNING: packages/ directory not found
    echo Attempting to install from requirements.txt (requires internet)...
    pip install -r requirements.txt
)

echo.
echo [4/5] Setting up model cache...
if exist ..\model_cache\models--google--translategemma-4b-it (
    echo Copying model cache...
    set CACHE_DIR=%USERPROFILE%\.cache\huggingface\hub
    if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
    xcopy /E /I /Y "..\model_cache\models--google--translategemma-4b-it" "%CACHE_DIR%\models--google--translategemma-4b-it"
    echo ✓ Model cache installed
) else (
    echo WARNING: Model cache not found in package
    echo You will need to download the model when you have internet
    echo Run: python download_model.py
)

echo.
echo [5/5] Creating .env file...
if not exist .env (
    (
        echo # Offline mode
        echo HF_HUB_OFFLINE=true
        echo TRANSFORMERS_OFFLINE=true
        echo.
        echo # GPU settings for NVIDIA A4000 16GB
        echo # CUDA_VISIBLE_DEVICES=0
    ) > .env
    echo ✓ .env file created
) else (
    echo .env file already exists
)

echo.
echo Checking GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo GPU DETECTED!
    echo ========================================
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo.
    echo If you have NVIDIA A4000 16GB, run setup_gpu.bat for optimal configuration
    echo.
)

echo.
echo ========================================
echo Setup completed!
echo ========================================
echo.
echo To start the server:
echo   1. cd backend
echo   2. .my-env\Scripts\activate
echo   3. python main.py
echo.
echo Or run: run_offline.bat
echo.
pause
