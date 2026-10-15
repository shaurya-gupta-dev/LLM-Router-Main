"""Runner script for LLM Router."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent

if not (BASE_DIR / ".env").exists() and (BASE_DIR / ".env.example").exists():
    shutil.copy(BASE_DIR / ".env.example", BASE_DIR / ".env")
    print("[INFO] Created .env from .env.example. Add your free API keys to .env to activate providers.")

from app.core.config import config

def main():
    host = config.settings.router_host
    port = config.settings.router_port
    print(f"\n=======================================================")
    print(f"  Starting LLM Router on http://{host}:{port}")
    print(f"  Web Dashboard: http://localhost:{port}")
    print(f"=======================================================\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
