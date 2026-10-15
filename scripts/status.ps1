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
Write-Host "=======================================================" -ForegroundColor Cyan
