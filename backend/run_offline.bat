@echo off
REM Script to run server in offline mode
REM Make sure you've downloaded the model first using: python download_model.py

echo ========================================
echo Starting server in OFFLINE mode
echo ========================================
echo.

REM Set offline mode environment variables
set HF_HUB_OFFLINE=true
set TRANSFORMERS_OFFLINE=true

REM Activate virtual environment and run server
cd /d %~dp0
call .my-env\Scripts\activate.bat
python main.py

pause
