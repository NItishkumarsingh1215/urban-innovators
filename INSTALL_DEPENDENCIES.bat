@echo off
setlocal enabledelayedexpansion
title BEL Urban Intelligence - Dependency Installer
color 0A

echo ==============================================================================
echo   🛰️ BHARAT ELECTRONICS LIMITED (BEL) - SIH 26124
echo   ONE-CLICK DEPENDENCY INSTALLER & ENVIRONMENT SETUP
echo ==============================================================================
echo.

cd /d "%~dp0"

set "HOST_PY="
where python >nul 2>&1 && set "HOST_PY=python"
if "!HOST_PY!"=="" (
    where py >nul 2>&1 && set "HOST_PY=py -3"
)

if "!HOST_PY!"=="" (
    echo [ERROR] Python not found on this system.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Using Python: !HOST_PY!

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [INFO] Creating clean virtual environment in .\venv...
    !HOST_PY! -m venv "%~dp0venv"
)

echo [INFO] Upgrading pip...
"%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip

echo [INFO] Installing required libraries from requirements.txt...
"%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"

echo.
echo ==============================================================================
echo   ✅ ALL DEPENDENCIES SUCCESSFULLY INSTALLED!
echo   You can now launch the app anytime by double-clicking RUN_PORTABLE.bat
echo ==============================================================================
echo.
pause
