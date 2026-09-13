@echo off
setlocal enabledelayedexpansion
title BEL Urban Intelligence Platform - SIH 26124 (Portable Launcher)
color 0B

echo ==============================================================================
echo   🛰️ BHARAT ELECTRONICS LIMITED (BEL) - SIH 26124
echo   AI-POWERED MOBILE URBAN INTELLIGENCE PLATFORM (PORTABLE LAUNCHER)
echo ==============================================================================
echo.

cd /d "%~dp0"
set "APP_DIR=%~dp0"

echo [1/3] Detecting Python Environment on this laptop...

set "USE_VENV=0"
set "PY_CMD="

:: Check if local venv exists and test if its python works
if exist "%APP_DIR%venv\Scripts\python.exe" (
    "%APP_DIR%venv\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Found working local virtual environment!
        set "USE_VENV=1"
        set "PY_CMD=%APP_DIR%venv\Scripts\python.exe"
    ) else (
        echo [INFO] Detected migrated venv. Auto-repairing paths for this laptop...
        python "%APP_DIR%tools\fix_portable_venv.py" >nul 2>&1
        "%APP_DIR%venv\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>&1
        if !errorlevel! equ 0 (
            echo [SUCCESS] Virtual environment repaired successfully!
            set "USE_VENV=1"
            set "PY_CMD=%APP_DIR%venv\Scripts\python.exe"
        )
    )
)

:: If venv is not ready, detect system Python
if !USE_VENV! equ 0 (
    set "HOST_PY="
    where python >nul 2>&1 && set "HOST_PY=python"
    if "!HOST_PY!"=="" (
        where py >nul 2>&1 && set "HOST_PY=py -3"
    )

    if "!HOST_PY!"=="" (
        echo [ERROR] Python was not found in PATH on this laptop.
        echo.
        echo Please ensure Python 3.9+ is installed (from https://www.python.org/downloads/)
        echo and check 'Add python.exe to PATH' during installation.
        echo.
        pause
        exit /b 1
    )

    echo [INFO] Host Python found: !HOST_PY!
    echo [INFO] Setting up self-contained portable virtualenv...
    !HOST_PY! -m venv "%APP_DIR%venv"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to initialize virtual environment.
        pause
        exit /b 1
    )

    echo [INFO] Installing required libraries into portable venv (one-time setup)...
    "%APP_DIR%venv\Scripts\python.exe" -m pip install --upgrade pip
    "%APP_DIR%venv\Scripts\python.exe" -m pip install -r "%APP_DIR%requirements.txt"
    set "PY_CMD=%APP_DIR%venv\Scripts\python.exe"
)

echo.
echo [2/3] Checking Core Packages...
"!PY_CMD!" -c "import streamlit, pandas; print('Streamlit & Data Engines Ready: v' + streamlit.__version__)"
if !errorlevel! neq 0 (
    echo [INFO] Installing Streamlit and core dependencies...
    "!PY_CMD!" -m pip install -r "%APP_DIR%requirements.txt"
)

echo.
echo [3/3] Launching BEL Urban Intelligence Platform...
echo ==============================================================================
echo   🌐 Dashboard URL: http://localhost:8501
echo   Opening in your default browser...
echo   (Press Ctrl+C in this console window to stop the platform)
echo ==============================================================================
echo.

:: Automatically trigger browser open after 2 seconds in the background
start /min cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8501"

:: Run Streamlit app
"!PY_CMD!" -m streamlit run "%APP_DIR%app.py" --server.port=8501 --server.headless=false

if !errorlevel! neq 0 (
    echo.
    echo [NOTICE] Streamlit server stopped.
    pause
)
