@echo off
title AI Watermark Remover (Image & Video)
cd /d %~dp0

echo ============================================================
echo   AI Watermark Remover & Metadata Sanitizer (Nano Banana)
echo ============================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python이 시스템에 설치되어 있지 않거나 PATH에 등록되지 않았습니다!
    echo https://www.python.org/downloads/ 에서 Python 3.10 또는 3.11을 설치하고,
    echo 설치 시 'Add python.exe to PATH' 체크박스를 반드시 선택해 주세요.
    echo.
    pause
    exit /b 1
)

:: 2. Setup Virtual Environment if missing
if not exist venv\Scripts\activate.bat (
    echo [1/4] 가상환경(venv)을 생성하는 중입니다...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] 가상환경 생성에 실패했습니다.
        pause
        exit /b 1
    )
    
    call venv\Scripts\activate.bat
    echo [2/4] 필수 패키지 및 PyTorch CUDA를 설치하는 중입니다 (최초 1회만 실행)...
    python -m pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    
    echo [3/4] AI 모델 가중치(ProPainter + LaMa)를 확인 및 다운로드하는 중입니다...
    python download_weights.py
    
    echo [4/4] 바탕화면 바로가기 아이콘을 생성하는 중입니다...
    python create_desktop_shortcut.py
) else (
    call venv\Scripts\activate.bat
)

:: 3. Launch App
echo.
echo ============================================================
echo   🚀 웹 UI를 실행합니다... (브라우저가 자동으로 열립니다)
echo ============================================================
echo.
venv\Scripts\python.exe app.py

if %errorlevel% neq 0 (
    echo.
    echo [오류] 프로그램 실행 중 문제가 발생했습니다. 위의 로그를 확인해 주세요.
    pause
)
