@echo off
title LLM Router - Unified Multi-Provider Gateway
echo Starting LLM Router...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run: python -m venv .venv
    pause
    exit /b 1
)
.venv\Scripts\python.exe run.py
pause
