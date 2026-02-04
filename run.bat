@echo off
REM ============================================================
REM Video Captioner - Quick Launch Script
REM ============================================================
REM Use this after you've run START_HERE.bat once
REM This skips the setup and launches the app immediately
REM ============================================================

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found!
    echo Please run START_HERE.bat first to complete setup.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Launch application
python captioner_compact.py

REM Handle errors
if %errorlevel% neq 0 (
    echo.
    echo Application closed with an error.
    echo Check %%TEMP%%\videocaptioner_debug.log for details.
    echo.
    pause
)
