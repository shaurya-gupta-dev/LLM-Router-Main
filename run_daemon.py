"""Daemon runner for LLM Router with PID tracking and detached logging."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
PID_FILE = BASE_DIR / "router.pid"
LOG_FILE = BASE_DIR / "router.log"

def cleanup_pid():
    try:
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def setup_daemon_logging():
    global LOG_FILE
    if sys.stdout is None:
        sys.stdout = open(str(LOG_FILE), "a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = open(str(LOG_FILE), "a", encoding="utf-8", buffering=1)

    handlers = [
        logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="a"),
    ]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

def main():
    setup_daemon_logging()
    logger = logging.getLogger("llm_router.daemon")

    # Record PID
    current_pid = os.getpid()
    PID_FILE.write_text(str(current_pid), encoding="utf-8")
    atexit.register(cleanup_pid)

    logger.info("=======================================================")
    logger.info("  LLM Router Daemon Starting (PID: %d)", current_pid)
    logger.info("  Log file: %s", LOG_FILE)
    logger.info("=======================================================")

    from app.core.config import config
    host = config.settings.router_host
    port = config.settings.router_port

    def handle_signal(sig, frame):
        logger.info("Received signal %s. Shutting down daemon...", sig)
        cleanup_pid()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass

    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True,
        )
    finally:
        cleanup_pid()
        logger.info("LLM Router Daemon Stopped.")

if __name__ == "__main__":
    main()
