# LLM Router - Startup Installer
param()

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$startupFolder = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupFolder 'LLM-Router-Background.lnk'
$targetPath = Join-Path $root 'start-background.vbs'

$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "wscript.exe"
$shortcut.Arguments = "`"$targetPath`""
$shortcut.WorkingDirectory = $root
$shortcut.Description = "LLM Router Background Daemon"
$shortcut.Save()

Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  >> Successfully installed LLM Router to Windows Startup!" -ForegroundColor Green
Write-Host ("  Shortcut: " + $shortcutPath) -ForegroundColor Cyan
Write-Host "  The router will now start silently whenever your PC boots." -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor Green
