@echo off
TITLE LocalDeepL - Installer
REM Elevate privileges if necessary to create shortcuts or install winget packages.
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~0' -Verb RunAs"
    exit /b
)

echo Starting installation...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
pause
