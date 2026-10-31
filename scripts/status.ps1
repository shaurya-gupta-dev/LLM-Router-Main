# LLM Router - Daemon Status Checker
param()

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  LLM Router - Background Daemon Status" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

$pidVal = "Unknown"
$pidFile = Join-Path $root "router.pid"
if (Test-Path $pidFile) {
    $pidVal = (Get-Content $pidFile -Raw).Trim()
}

$startupLnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'LLM-Router-Background.lnk'
$autoStartEnabled = (Test-Path $startupLnk)

try {
    $r = Invoke-RestMethod -Uri "http://localhost:8080/api/health" -TimeoutSec 3
    Write-Host "  Status:             [RUNNING] (Online)" -ForegroundColor Green
    Write-Host ("  Process PID:        " + $pidVal) -ForegroundColor White
    Write-Host ("  Providers Active:   " + $r.providers_configured) -ForegroundColor White
    Write-Host ("  Models Configured:  " + $r.models_configured) -ForegroundColor White
    Write-Host "  Endpoint:           http://localhost:8080" -ForegroundColor White
    Write-Host "  Dashboard:          http://localhost:8080/" -ForegroundColor White
} catch {
    Write-Host "  Status:             [STOPPED] (Offline)" -ForegroundColor Red
    Write-Host "  To start:           Run start-background.bat" -ForegroundColor Yellow
}

if ($autoStartEnabled) {
    Write-Host "  Auto-Start on Boot: [ENABLED] (Windows Startup)" -ForegroundColor Green
} else {
    Write-Host "  Auto-Start on Boot: [DISABLED] (Run install-startup.bat to enable)" -ForegroundColor Yellow
}

Write-Host "=======================================================" -ForegroundColor Cyan
