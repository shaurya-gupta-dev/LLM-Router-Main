@echo off
title LLM Router - Auto-Start Uninstaller
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\uninstall-startup.ps1"
pause

