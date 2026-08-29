@echo off
title Video Watermark Remover (ProPainter + Gradio)
cd /d %~dp0

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo Starting Watermark Remover Web UI...
python app.py
pause
