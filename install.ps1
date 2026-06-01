$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "======================================================="
Write-Host "Installing Local LLM PDF OCR Dependencies"
Write-Host "======================================================="

# 1. Check/Install uv
if (!(Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    # Add uv to the current process PATH so the script can continue
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    Write-Host "uv is already installed."
}

# 2. Sync dependencies
Write-Host "`nSyncing python dependencies with uv..."
Set-Location -Path $ScriptDir
# uv will automatically download the correct python version based on .python-version if it is missing
uv sync --extra web

# 3. Check Docker
Write-Host "`nChecking for Docker (required for Redis)..."
if (!(Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: Docker is not installed or not in PATH." -ForegroundColor Yellow
    Write-Host "Docker is required to run Redis for the translation features." -ForegroundColor Yellow
    Write-Host "Please install Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
} else {
    Write-Host "Docker is installed."
}

# 4. Create Shortcuts
Write-Host "`nCreating shortcuts..."

$WshShell = New-Object -comObject WScript.Shell

# Desktop Shortcut
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path -Path $DesktopPath -ChildPath "Local LLM PDF OCR.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = Join-Path -Path $ScriptDir -ChildPath "start_app.vbs"
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.IconLocation = "%SystemRoot%\system32\SHELL32.dll,22"
$Shortcut.Save()
Write-Host "Created Desktop Shortcut: $ShortcutPath"

# Start Menu Shortcut
$StartMenuPath = [Environment]::GetFolderPath("Programs")
$ShortcutPathSM = Join-Path -Path $StartMenuPath -ChildPath "Local LLM PDF OCR.lnk"
$ShortcutSM = $WshShell.CreateShortcut($ShortcutPathSM)
$ShortcutSM.TargetPath = Join-Path -Path $ScriptDir -ChildPath "start_app.vbs"
$ShortcutSM.WorkingDirectory = $ScriptDir
$ShortcutSM.IconLocation = "%SystemRoot%\system32\SHELL32.dll,22"
$ShortcutSM.Save()
Write-Host "Created Start Menu Shortcut: $ShortcutPathSM"

# Stop Shortcut (Start Menu)
$StopShortcutPathSM = Join-Path -Path $StartMenuPath -ChildPath "Stop Local LLM PDF OCR.lnk"
$StopShortcutSM = $WshShell.CreateShortcut($StopShortcutPathSM)
$StopShortcutSM.TargetPath = Join-Path -Path $ScriptDir -ChildPath "stop_app.bat"
$StopShortcutSM.WorkingDirectory = $ScriptDir
$StopShortcutSM.IconLocation = "%SystemRoot%\system32\SHELL32.dll,28" # Stop icon
$StopShortcutSM.Save()
Write-Host "Created Stop Shortcut in Start Menu: $StopShortcutPathSM"

Write-Host "`n======================================================="
Write-Host "Installation Complete! You can now run the app from your Desktop."
Write-Host "======================================================="
