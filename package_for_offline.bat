@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Package CSV Translator for offline deployment
REM  Fixes: proper error handling, robust copy, Vite base path
REM ============================================================
echo ========================================
echo Packaging for OFFLINE installation
echo ========================================
echo.

REM ---------- Package directory name ----------
set PACKAGE_NAME=csv-translator-offline
REM Use a safe date format (YYYYMMDD)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set DATESTAMP=%DT:~0,8%
set PACKAGE_DIR=%PACKAGE_NAME%-%DATESTAMP%

echo Creating package directory: %PACKAGE_DIR%
if exist "%PACKAGE_DIR%" rmdir /s /q "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%\packages"
mkdir "%PACKAGE_DIR%\model_cache"

REM ============================================================
echo.
echo [1/7] Copying source code...
echo.
xcopy /E /I /Y "backend" "%PACKAGE_DIR%\backend"
xcopy /E /I /Y "frontend" "%PACKAGE_DIR%\frontend"
echo.
echo OK - source code copied
echo.

REM ============================================================
echo [2/7] Building frontend (with base ./ for offline)...
echo.
if exist "frontend\package.json" (
    pushd frontend
    if not exist node_modules (
        echo Installing npm dependencies first...
        call npm install
    )
    echo Running vite build...
    call npm run build
    popd

    if exist "frontend\dist\index.html" (
        xcopy /E /I /Y "frontend\dist" "%PACKAGE_DIR%\frontend_dist"
        echo.
        echo OK - frontend_dist created
    ) else (
        echo WARNING: frontend\dist\index.html not found after build!
    )
) else (
    echo WARNING: frontend\package.json not found, skipping frontend build
)
echo.

REM ============================================================
echo [3/7] Exporting Python packages...
echo.

set PIP_EXE=backend\.my-env\Scripts\pip.exe
set PY_EXE=backend\.my-env\Scripts\python.exe

if not exist "%PY_EXE%" (
    echo WARNING: %PY_EXE% not found. Skipping package download.
    echo Create venv first: python -m venv backend\.my-env
    goto :skip_packages
)

REM ---- freeze full list ----
echo Exporting pip freeze...
"%PY_EXE%" -m pip freeze > "%PACKAGE_DIR%\requirements-full.txt"
echo   Wrote requirements-full.txt

REM ---- download wheels (skip torch in requirements.txt, download separately) ----
echo.
echo Downloading wheels from requirements.txt (excluding torch)...

REM Create a temp requirements without torch lines
findstr /v /i "^torch" "backend\requirements.txt" > "%TEMP%\req_no_torch.txt"
"%PIP_EXE%" download -r "%TEMP%\req_no_torch.txt" -d "%PACKAGE_DIR%\packages"
if !ERRORLEVEL! NEQ 0 (
    echo WARNING: Some packages from requirements.txt failed to download.
)
del "%TEMP%\req_no_torch.txt" 2>nul

echo.
echo Downloading PyTorch CUDA 12.4 wheels (compatible with CUDA 12.7)...
"%PIP_EXE%" download torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 -d "%PACKAGE_DIR%\packages" --index-url https://download.pytorch.org/whl/cu124
if !ERRORLEVEL! NEQ 0 (
    echo WARNING: CUDA wheels failed. Downloading CPU wheels as fallback...
    "%PIP_EXE%" download torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -d "%PACKAGE_DIR%\packages"
)

REM ---- also grab python-dotenv (used by main.py) ----
"%PIP_EXE%" download python-dotenv -d "%PACKAGE_DIR%\packages" >nul 2>&1

echo.
echo OK - packages download step finished
echo.

:skip_packages

REM ============================================================
echo [4/7] Copying model cache...
echo.

set "HF_CACHE=%USERPROFILE%\.cache\huggingface\hub"
set "MODEL_SRC=%HF_CACHE%\models--google--translategemma-4b-it"

if exist "%MODEL_SRC%" (
    echo Source: %MODEL_SRC%
    echo This may take a while... model is ~8-10 GB

    REM Use robocopy for reliable large-dir copy (exit code 0-7 = OK)
    robocopy "%MODEL_SRC%" "%PACKAGE_DIR%\model_cache\models--google--translategemma-4b-it" /E /NP /NFL /NDL /NJH /NJS
    if !ERRORLEVEL! LSS 8 (
        echo OK - model cache copied
    ) else (
        echo WARNING: robocopy reported errors. Falling back to xcopy...
        xcopy /E /I /Y "%MODEL_SRC%" "%PACKAGE_DIR%\model_cache\models--google--translategemma-4b-it"
    )
) else (
    echo WARNING: Model cache not found at %MODEL_SRC%
    echo Please run:  cd backend ^& python download_model.py  first
    echo Creating placeholder...
    echo Model cache not found. Download model first then re-run packaging. > "%PACKAGE_DIR%\model_cache\README.txt"
)
echo.

REM ============================================================
echo [5/7] Copying setup and run scripts...
echo.

REM setup.bat  (goes to package ROOT, runs from backend context)
if exist "backend\setup_offline.bat" (
    copy /Y "backend\setup_offline.bat" "%PACKAGE_DIR%\setup.bat" >nul
    echo   setup.bat  -- copied
) else (
    echo   setup_offline.bat not found, creating minimal setup.bat...
    (
        echo @echo off
        echo echo Run: cd backend ^&^& python -m venv .my-env ^&^& .my-env\Scripts\pip install --no-index --find-links=..\packages -r requirements.txt
        echo pause
    ) > "%PACKAGE_DIR%\setup.bat"
    echo   setup.bat  -- created (minimal)
)

REM INSTALL.md
if exist "backend\INSTALL_OFFLINE.md" (
    copy /Y "backend\INSTALL_OFFLINE.md" "%PACKAGE_DIR%\INSTALL.md" >nul
    echo   INSTALL.md -- copied
) else (
    echo   INSTALL_OFFLINE.md not found, skipped
)

REM run_offline.bat inside backend/
if exist "backend\run_offline.bat" (
    copy /Y "backend\run_offline.bat" "%PACKAGE_DIR%\backend\run_offline.bat" >nul
    echo   run_offline.bat -- copied into backend/
)

REM GPU helpers
if exist "backend\setup_gpu.bat" (
    copy /Y "backend\setup_gpu.bat" "%PACKAGE_DIR%\backend\setup_gpu.bat" >nul
    echo   setup_gpu.bat  -- copied into backend/
)
if exist "backend\GPU_SETUP_A4000.md" (
    copy /Y "backend\GPU_SETUP_A4000.md" "%PACKAGE_DIR%\GPU_SETUP.md" >nul
    echo   GPU_SETUP.md   -- copied
)
echo.

REM ============================================================
echo [6/7] Creating PACKAGE_INFO.txt...
echo.

(
    echo Package created: %DATE% %TIME%
    echo Source machine : %COMPUTERNAME%
    echo Date stamp     : %DATESTAMP%
    echo.
    echo Contents:
    echo   backend/           - Python source code
    echo   frontend/          - React source code
    echo   frontend_dist/     - Pre-built frontend (open index.html)
    echo   packages/          - Python wheels for offline pip install
    echo   model_cache/       - Hugging Face model cache (~10GB)
    echo   requirements-full.txt - Full pip freeze list
    echo   setup.bat          - Offline installation script
    echo   INSTALL.md         - Installation instructions
    echo   GPU_SETUP.md       - GPU optimization guide
) > "%PACKAGE_DIR%\PACKAGE_INFO.txt"
type "%PACKAGE_DIR%\PACKAGE_INFO.txt"
echo.

REM ============================================================
echo [7/7] Creating README.txt...
echo.

(
    echo ========================================
    echo  CSV Translator - Offline Package
    echo ========================================
    echo.
    echo This package contains everything needed to run
    echo CSV Translator on a machine WITHOUT internet.
    echo.
    echo QUICK START:
    echo   1. Extract this folder to your target machine
    echo   2. Run setup.bat  (creates venv + installs packages + copies model)
    echo   3. cd backend
    echo      .my-env\Scripts\activate
    echo      python main.py
    echo   4. Open frontend_dist\index.html in a browser
    echo      (or host frontend_dist/ with any static file server)
    echo.
    echo   If backend is NOT on localhost:8000, edit
    echo   frontend_dist\config.json  and change "apiUrl".
    echo   No rebuild needed!
    echo.
    echo See INSTALL.md for detailed instructions.
) > "%PACKAGE_DIR%\README.txt"
echo   README.txt created

REM ============================================================
echo.
echo ========================================
echo  DONE! Package: %PACKAGE_DIR%
echo ========================================
echo.
echo TARGET MACHINE:
echo   GPU  : NVIDIA A4000
echo   VRAM : 16GB
echo   CUDA : 12.7
echo.
echo Next steps:
echo   1. Compress  %PACKAGE_DIR%  to .rar or .zip
echo   2. Copy to target machine
echo   3. Extract and run:  setup.bat
echo   4. For GPU optimization: backend\setup_gpu.bat
echo.

echo Package contents:
echo -------
dir "%PACKAGE_DIR%" /b
echo -------
echo.
echo Package size:
for /f "tokens=3" %%A in ('dir "%PACKAGE_DIR%" /s ^| findstr /c:"File(s)"') do (
    echo   Total: %%A bytes
)
echo.
pause
endlocal
