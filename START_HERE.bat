@echo off
REM ============================================================
REM Video Captioner - Automated Setup and Launch Script
REM ============================================================

setlocal EnableDelayedExpansion

REM Setup logging
set "LOG_FILE=%~dp0setup_log.txt"
echo Setup started at %date% %time% > "%LOG_FILE%"
echo Setup started at %date% %time%
echo.

cd /d "%~dp0"

echo ============================================================
echo           Video Captioner - Automated Setup
echo ============================================================
echo.
echo Working directory: %CD%
echo.
echo ============================================================
echo [CHECK] System diagnostics...
echo ============================================================
echo.

REM Check for Python 3.12 or 3.11 in common locations first
set "PYTHON_CMD="
set "PY_VER="

REM Try Python 3.12 in standard location
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    for /f "tokens=2" %%v in ('"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --version 2^>^&1') do set PY_VER=%%v
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    echo [OK] Python 3.12 found: !PY_VER!
    goto version_ok
)

REM Try Python 3.11 in standard location
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    for /f "tokens=2" %%v in ('"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" --version 2^>^&1') do set PY_VER=%%v
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    echo [OK] Python 3.11 found: !PY_VER!
    goto version_ok
)

REM Try python command in PATH
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
    
    REM Check if it's 3.11 or 3.12
    echo !PY_VER! | findstr /r "^3\.11\." >nul
    if !errorlevel! == 0 (
        set "PYTHON_CMD=python"
        echo [OK] Python found: !PY_VER!
        goto version_ok
    )
    echo !PY_VER! | findstr /r "^3\.12\." >nul
    if !errorlevel! == 0 (
        set "PYTHON_CMD=python"
        echo [OK] Python found: !PY_VER!
        goto version_ok
    )
    
    echo [WARNING] Python !PY_VER! found in PATH, but need 3.11 or 3.12
)

REM Python 3.11/3.12 not found
echo [ERROR] Python 3.11 or 3.12 not found!
echo.
echo Checked:
echo  - %LOCALAPPDATA%\Programs\Python\Python312
echo  - %LOCALAPPDATA%\Programs\Python\Python311
echo  - python command in PATH
echo.
echo Python 3.13+ is TOO NEW - PyTorch 2.8.0 wheels are not available for it.
echo Python 3.10 or older is TOO OLD.
echo.
echo Please install Python 3.12 from:
echo https://www.python.org/downloads/
echo.
echo IMPORTANT: During installation, check "Add Python to PATH"
echo.
pause
exit /b 1

:version_ok
echo.

REM Check for venv
echo [CHECK] Virtual environment...
if exist "venv\" (
    echo [OK] Virtual environment exists
) else (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Check FFmpeg
echo [CHECK] FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] FFmpeg found
) else (
    if exist "ffmpeg.exe" (
        echo [OK] Local ffmpeg.exe found
    ) else (
        echo [INFO] FFmpeg not found - will download later
    )
)
echo.

echo ============================================================
echo           Starting setup process...
echo ============================================================
echo.

REM Activate venv
echo [1/5] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if %errorlevel% == 0 (
    echo [OK] pip upgraded
) else (
    echo [WARNING] pip upgrade failed, continuing...
)
echo.

REM Install dependencies
echo [3/5] Installing dependencies...
echo This may take 5-10 minutes on first run...
echo Progress is being logged to: %LOG_FILE%
echo.

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    echo Make sure you have all files from GitHub.
    echo.
    pause
    exit /b 1
)

pip show torch >nul 2>&1
if %errorlevel% == 0 goto update_deps
goto install_deps

:update_deps
echo Dependencies already installed, checking for updates...
pip install -r requirements.txt --upgrade >> "%LOG_FILE%" 2>&1
goto check_install

:install_deps
echo Installing all dependencies (please wait)...
pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
goto check_install

:check_install
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    echo Check log file: %LOG_FILE%
    echo.
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Download FFmpeg if needed
echo [4/5] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] FFmpeg available
) else (
    if exist "ffmpeg.exe" (
        echo [OK] Local ffmpeg.exe found
    ) else (
        if exist "scripts\download_ffmpeg.py" (
            echo Downloading FFmpeg...
            python scripts\download_ffmpeg.py >> "%LOG_FILE%" 2>&1
            if not !errorlevel! == 0 (
                echo [WARNING] FFmpeg download failed
                echo You may need to install it manually: winget install ffmpeg
            ) else (
                echo [OK] FFmpeg downloaded
            )
        ) else (
            echo [WARNING] FFmpeg not found
            echo Install manually: winget install ffmpeg
        )
    )
)
echo.

REM Launch application
echo [5/5] Launching application...
echo.

if not exist "captioner_compact.py" (
    echo [ERROR] captioner_compact.py not found!
    echo Make sure you have all files from GitHub.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo           Setup Complete! Starting Video Captioner...
echo ============================================================
echo.
echo TIP: Next time, use run.bat for faster startup!
echo.

python captioner_compact.py

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo Application closed with an error
    echo ============================================================
    echo.
    echo Troubleshooting:
    echo  1. Install Visual C++ Redistributable:
    echo     https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo  2. Check log: %%TEMP%%\videocaptioner_debug.log
    echo  3. Check setup log: %LOG_FILE%
    echo.
    pause
)

endlocal
