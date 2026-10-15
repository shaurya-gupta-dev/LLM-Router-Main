# PowerShell Launcher for LLM Router
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Starting LLM Router..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at $pythonExe" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

& $pythonExe (Join-Path $PSScriptRoot "run.py")
