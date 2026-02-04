@echo off
REM ============================================================
REM Video Captioner - Automated Setup and Launch Script
REM ============================================================
REM This script will:
REM   1. Install Python 3.11 if not found
REM   2. Create a virtual environment
REM   3. Install all dependencies
REM   4. Download FFmpeg
REM   5. Launch the Video Captioner application
REM
REM Just double-click this file to get started!
REM ============================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo           Video Captioner - Automated Setup
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM ============================================================
REM STEP 1: Check for Python 3.11
REM ============================================================

echo [1/5] Checking for Python 3.11...

REM Try to find Python in PATH
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
    echo Found Python !PYTHON_VER!
    
    REM Check if it's Python 3.11.x
    echo !PYTHON_VER! | findstr /r "3\.11\." >nul
    if !errorlevel! == 0 (
        set PYTHON_CMD=python
        goto :python_found
    ) else (
        echo Warning: Found Python !PYTHON_VER!, but need Python 3.11.x
    )
)

REM Try python3 command
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2" %%v in ('python3 --version 2^>^&1') do set PYTHON_VER=%%v
    echo Found Python !PYTHON_VER!
    
    echo !PYTHON_VER! | findstr /r "3\.11\." >nul
    if !errorlevel! == 0 (
        set PYTHON_CMD=python3
        goto :python_found
    )
)

REM Try common installation paths
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
) do (
    if exist %%p (
        echo Found Python at %%p
        set PYTHON_CMD=%%p
        goto :python_found
    )
)

REM Python not found - install it
echo Python 3.11 not found. Installing automatically...
echo.
echo This will use Windows Package Manager (winget) to install Python.
echo Please wait, this may take a few minutes...
echo.

REM Check if winget is available
winget --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Windows Package Manager (winget) not found.
    echo.
    echo Please install Python 3.11 manually:
    echo   1. Visit: https://www.python.org/downloads/
    echo   2. Download Python 3.11.x
    echo   3. Run the installer
    echo   4. Make sure to check "Add Python to PATH"
    echo   5. Run this script again
    echo.
    pause
    exit /b 1
)

REM Install Python using winget
echo Installing Python 3.11...
winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements

if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python automatically.
    echo Please install Python 3.11 manually from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Python installed successfully!
echo Please close this window and run START_HERE.bat again.
echo.
pause
exit /b 0

:python_found
echo [OK] Python found: %PYTHON_CMD%
echo.

REM ============================================================
REM STEP 2: Check/Create Virtual Environment
REM ============================================================

echo [2/5] Setting up virtual environment...

if exist "venv\" (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM ============================================================
REM STEP 3: Install/Upgrade pip
REM ============================================================

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.
echo.

REM ============================================================
REM STEP 4: Install Dependencies
REM ============================================================

echo [4/5] Installing dependencies...
echo This may take 5-10 minutes on first run (downloading PyTorch)...
echo Please be patient...
echo.

REM Check if requirements are already installed
pip show torch >nul 2>&1
if %errorlevel% == 0 (
    echo Dependencies already installed. Checking for updates...
    pip install -r requirements.txt --upgrade --quiet
) else (
    echo Installing all dependencies (this will take a while)...
    pip install -r requirements.txt
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    echo.
    echo Try running this manually:
    echo   1. Open Command Prompt
    echo   2. Navigate to this folder
    echo   3. Run: venv\Scripts\activate
    echo   4. Run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [OK] All dependencies installed.
echo.

REM ============================================================
REM STEP 5: Download FFmpeg
REM ============================================================

echo [5/5] Checking for FFmpeg...

if exist "ffmpeg.exe" (
    echo FFmpeg already exists.
) else (
    echo Downloading FFmpeg...
    python scripts\download_ffmpeg.py
    
    if !errorlevel! neq 0 (
        echo WARNING: Could not download FFmpeg automatically.
        echo The application will still work if FFmpeg is installed system-wide.
        echo.
        echo To install FFmpeg manually:
        echo   1. Run: winget install ffmpeg
        echo   OR
        echo   2. Download from: https://ffmpeg.org/download.html
        echo.
    ) else (
        echo [OK] FFmpeg downloaded.
    )
)
echo.

REM ============================================================
REM LAUNCH APPLICATION
REM ============================================================

echo ============================================================
echo           Setup Complete! Launching Application...
echo ============================================================
echo.
echo TIP: Next time, you can use run.bat for faster startup!
echo.

timeout /t 2 >nul

REM Launch the application
python captioner_compact.py

REM If application exits with error
if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo Application closed with an error.
    echo.
    echo Troubleshooting:
    echo   1. Check that Visual C++ Redistributable is installed
    echo      Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo   2. Check the log file:
    echo      %%TEMP%%\videocaptioner_debug.log
    echo.
    echo   3. Make sure all files were downloaded from GitHub
    echo ============================================================
    echo.
    pause
)

endlocal
