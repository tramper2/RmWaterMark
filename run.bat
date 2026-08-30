@echo off
chcp 65001 >nul
title AI Watermark Remover - Image and Video
cd /d "%~dp0"

echo ============================================================
echo   AI Watermark Remover + Metadata Sanitizer (Nano Banana)
echo ============================================================
echo.

REM 1. Check Python installation
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH!
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo Make sure to check 'Add python.exe to PATH' during installation.
    echo.
    pause
    exit /b 1
)

REM 2. Check if Virtual Environment exists
if exist "venv\Scripts\activate.bat" goto RUN_APP

:CREATE_VENV
echo [1/4] Creating virtual environment (venv)...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo [2/4] Installing PyTorch CUDA and dependencies (first-time setup)...
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo [3/4] Checking and downloading AI model weights...
python download_weights.py

echo [4/4] Creating desktop shortcut icon...
python create_desktop_shortcut.py

:RUN_APP
call venv\Scripts\activate.bat
echo.
echo ============================================================
echo   Starting Web UI... (Browser will open automatically)
echo ============================================================
echo.
venv\Scripts\python.exe app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error. Please check the log above.
    pause
)
