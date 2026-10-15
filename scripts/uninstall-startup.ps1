# LLM Router - Startup Uninstaller
param()

$startupFolder = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupFolder 'LLM-Router-Background.lnk'

if (Test-Path $shortcutPath) {
    Remove-Item $shortcutPath -Force
    Write-Host "=======================================================" -ForegroundColor Green
    Write-Host "  >> LLM Router removed from Windows Startup." -ForegroundColor Green
    Write-Host "  The router will no longer start automatically on boot." -ForegroundColor White
    Write-Host "=======================================================" -ForegroundColor Green
} else {
    Write-Host ">> Startup shortcut not found. Nothing to remove." -ForegroundColor Yellow
}
