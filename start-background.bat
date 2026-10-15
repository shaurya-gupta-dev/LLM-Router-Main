@echo off
cd /d "%~dp0"
echo Starting LLM Router in background...
cscript //nologo start-background.vbs
ping 127.0.0.1 -n 3 >nul
powershell -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8080/api/health' -TimeoutSec 5; Write-Host '>> LLM Router is RUNNING in background on http://localhost:8080' -ForegroundColor Green; if (Test-Path router.pid) { Write-Host ('   PID: ' + (Get-Content router.pid)) -ForegroundColor Cyan }; Start-Process 'http://localhost:8080' } catch { Write-Host '>> Warning: Server is still starting up. Check status with status-router.bat' -ForegroundColor Yellow }"
