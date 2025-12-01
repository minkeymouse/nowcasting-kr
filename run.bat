@echo off
setlocal

echo ==========================================
echo   Initializing Chatbot Environment...
echo ==========================================

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python 3.10+.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment (.venv) with uv...
uv venv .venv
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Installing dependencies from pyproject.toml...
uv pip install -r pyproject.toml
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install Python packages.
    pause
    exit /b 1
)

echo.
echo Starting Chatbot server...
start "" http://localhost:8000
uv run app/main.py

echo.
echo Chatbot stopped.
pause
endlocal
