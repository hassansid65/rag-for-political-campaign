@echo off
REM Start the backend using backend\.venv, creating it on first run.
setlocal
cd /d "%~dp0backend"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [setup] No virtualenv found. Creating backend\.venv with Python 3.10...
    python -m venv .venv
    if not exist ".venv\pyvenv.cfg" (
        echo [error] Virtualenv creation failed. Is Python 3.10 on PATH?
        python --version
        exit /b 1
    )
    "%PY%" -m pip install --upgrade pip
    echo [setup] Installing torch ^(CPU wheel^)...
    "%PY%" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [error] torch install failed. If this was an OSError on a .tmp file,
        echo         delete backend\.venv and run this script again.
        exit /b 1
    )
    echo [setup] Installing remaining requirements...
    "%PY%" -m pip install -r requirements.txt || exit /b 1
)

if not exist ".env" (
    echo [setup] Creating .env from .env.example - add your ANTHROPIC_API_KEY.
    copy /y ".env.example" ".env" >nul
)

echo.
"%PY%" scripts\verify_env.py
echo.
echo [run] Starting API on http://localhost:8000  ^(docs at /docs^)
"%PY%" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
endlocal
