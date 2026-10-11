<# : EvoLoRA one-click installer for Windows. Batch and PowerShell in one file: cmd runs
@echo off
setlocal
title EvoLoRA Setup
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Expression ([IO.File]::ReadAllText('%~f0'))"
echo.
pause
exit /b
#>
# ---------------------------------------------------------------------------------------
# EvoLoRA installer (PowerShell part). Installs for the current user only, no admin rights:
#   %LOCALAPPDATA%\EvoLoRA\app      EvoLoRA source (replaced on every run = update)
#   %LOCALAPPDATA%\EvoLoRA          models, tools, adapters, history (kept across updates)
# Sources: uv from astral.sh, EvoLoRA from GitHub. Models and training tools are fetched
# later from inside the app ("Install everything"), each from its official source.
# ---------------------------------------------------------------------------------------
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # makes Invoke-WebRequest much faster
$Repo   = 'AkshayLanghani/EvoLoRA'
$Branch = 'main'
$Root   = Join-Path $env:LOCALAPPDATA 'EvoLoRA'
$App    = Join-Path $Root 'app'

Write-Host ''
Write-Host '  EvoLoRA setup' -ForegroundColor Green
Write-Host "  Installing into $Root"
Write-Host ''
New-Item -ItemType Directory -Force $Root | Out-Null

# 1. uv: Astral's Python package manager (per-user install).
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host '[1/4] Installing uv (astral.sh)...'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
} else {
    Write-Host '[1/4] uv is already installed'
}
$UvExe = (Get-Command uv).Source

# 2. EvoLoRA source from GitHub.
Write-Host '[2/4] Downloading EvoLoRA...'
$Zip = Join-Path $env:TEMP 'evolora-src.zip'
$Tmp = Join-Path $env:TEMP 'evolora-src'
if ($env:EVOLORA_SOURCE_ZIP) {
    # For testing unreleased builds: install from a local source zip instead of GitHub.
    Copy-Item $env:EVOLORA_SOURCE_ZIP $Zip -Force
} else {
    Invoke-WebRequest "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $Zip -UseBasicParsing
}
if (Test-Path $Tmp) { Remove-Item $Tmp -Recurse -Force }
Expand-Archive $Zip $Tmp
$Src = Get-ChildItem $Tmp -Directory | Select-Object -First 1
Get-Process | Where-Object { $_.Path -like "$App*" } | Stop-Process -Force -ErrorAction SilentlyContinue
if (Test-Path $App) { Remove-Item $App -Recurse -Force }
Move-Item $Src.FullName $App
Remove-Item $Zip, $Tmp -Recurse -Force -ErrorAction SilentlyContinue

# 3. Python environment (uv downloads Python 3.11 itself if needed).
Write-Host '[3/4] Setting up Python (the first time takes a few minutes)...'
& $UvExe sync --project $App --no-dev
if ($LASTEXITCODE -ne 0) { throw 'Setting up the Python environment failed (uv sync).' }

# 4. Launcher, Start menu + desktop shortcuts, then start EvoLoRA.
$Launcher = Join-Path $Root 'EvoLoRA.cmd'
@"
@echo off
title EvoLoRA (close this window to stop)
cd /d "%~dp0"
"$UvExe" run --project "$App" --no-dev evolora agent
"@ | Set-Content $Launcher -Encoding ASCII

$Shell = New-Object -ComObject WScript.Shell
foreach ($Dir in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) {
    $Link = $Shell.CreateShortcut((Join-Path $Dir 'EvoLoRA.lnk'))
    $Link.TargetPath = $Launcher
    $Link.WorkingDirectory = $Root
    $Link.Description = 'EvoLoRA: local, auditable LoRA fine-tuning'
    $Link.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
    $Link.Save()
}

Write-Host '[4/4] Starting EvoLoRA...'
Start-Process $Launcher -WorkingDirectory $Root
Write-Host ''
Write-Host '  Done. EvoLoRA is starting and will open in your browser.' -ForegroundColor Green
Write-Host '  Next time, start it from the Start menu or the desktop shortcut "EvoLoRA".'
Write-Host '  On the EvoLoRA website, press Connect to use it from there.'
