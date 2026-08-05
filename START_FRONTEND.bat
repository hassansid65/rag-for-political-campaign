@echo off
setlocal
cd /d "%~dp0frontend"

REM Next 16 runs TypeScript checking in a separate worker that can hard-crash
REM (exit 3221226505 / 0xC0000409) on a memory-tight machine. Raising the heap
REM prevents it; observed on a box with ~2 GB free.
if not defined NODE_OPTIONS set "NODE_OPTIONS=--max-old-space-size=4096"

if not exist "node_modules" (
    echo [setup] Installing frontend dependencies...
    call npm install --no-fund --no-audit || exit /b 1
)

if not exist ".env.local" (
    echo [setup] Creating .env.local from .env.local.example
    copy /y ".env.local.example" ".env.local" >nul
)

echo [run] Starting UI on http://localhost:3000
call npm run dev
endlocal
