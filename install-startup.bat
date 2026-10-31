@echo off
title LLM Router - Auto-Start Installer
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-startup.ps1"
pause

