# LLM Router - Silent Background Daemon Launcher
param()

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    $pythonw = Join-Path $root ".venv\Scripts\python.exe"
}

if (-not (Test-Path $pythonw)) {
    Write-Host "[ERROR] Virtual environment not found at $pythonw" -ForegroundColor Red
    Write-Host "Please set up .venv first: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Check if already running
try {
    $existing = Invoke-RestMethod -Uri "http://localhost:8080/api/health" -TimeoutSec 2
    Write-Host "=======================================================" -ForegroundColor Green
    Write-Host "  >> LLM Router is ALREADY running in the background!" -ForegroundColor Green
    Write-Host "  Host:      http://localhost:8080" -ForegroundColor Cyan
    Write-Host "  Dashboard: http://localhost:8080/" -ForegroundColor Cyan
    if (Test-Path (Join-Path $root "router.pid")) {
        $pidNum = (Get-Content (Join-Path $root "router.pid") -Raw).Trim()
        Write-Host "  PID:       $pidNum" -ForegroundColor Cyan
    }
    Write-Host "=======================================================" -ForegroundColor Green
    exit 0
} catch {}

$daemonScript = Join-Path $root "run_daemon.py"

# Launch detached process without opening a console window
Start-Process -FilePath $pythonw -ArgumentList "`"$daemonScript`"" -WorkingDirectory $root -WindowStyle Hidden

Write-Host "Starting LLM Router daemon in background..." -ForegroundColor Cyan

$online = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:8080/api/health" -TimeoutSec 2
        $online = $true
        break
    } catch {
        Write-Host -NoNewline "."
    }
}
Write-Host ""

if ($online) {
    Write-Host "=======================================================" -ForegroundColor Green
    Write-Host "  >> LLM Router is RUNNING in the background!" -ForegroundColor Green
    Write-Host "  Host:      http://localhost:8080" -ForegroundColor Cyan
    Write-Host "  Dashboard: http://localhost:8080/" -ForegroundColor Cyan
    if (Test-Path (Join-Path $root "router.pid")) {
        $pidNum = (Get-Content (Join-Path $root "router.pid") -Raw).Trim()
        Write-Host "  PID:       $pidNum" -ForegroundColor Cyan
    }
    Write-Host "=======================================================" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Server took longer than expected to start." -ForegroundColor Yellow
    Write-Host "Check status with .\scripts\status.ps1 or view router.log" -ForegroundColor Yellow
}
