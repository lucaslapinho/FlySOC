@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the virtual environment with scripts\bootstrap.ps1 first.
    pause
    exit /b 1
)
echo FlySOC Research Platform - keep this window open. Ctrl+C stops the server.
".venv\Scripts\python.exe" scripts\brain_lab.py --open-browser
pause
