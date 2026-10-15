"""Main FastAPI application entry point for LLM Router."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.v1.admin import router as admin_router
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.messages import router as messages_router
from app.api.v1.models import router as models_router
from app.core.config import config
from app.core.key_pool import key_manager
from app.core.router import router_engine

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm_router")

TEMPLATES_DIR = Path(__file__).resolve().parent / "dashboard" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup banner
    logger.info("=" * 65)
    logger.info("  >> LLM ROUTER -- UNIFIED MULTI-PROVIDER GATEWAY")
    logger.info("=" * 65)
    logger.info("Host: http://%s:%d", config.settings.router_host, config.settings.router_port)
    logger.info("Dashboard: http://localhost:%d", config.settings.router_port)
    logger.info("Anthropic Endpoint: http://localhost:%d/v1/messages", config.settings.router_port)
    logger.info("OpenAI Endpoint: http://localhost:%d/v1/chat/completions", config.settings.router_port)
    logger.info("-" * 65)

    # Log provider key status
    active_providers = 0
    total_keys = 0
    for p_id in config.providers.keys():
        pool = key_manager.get_pool(p_id)
        if pool and pool.total_keys > 0:
            active_providers += 1
            total_keys += pool.total_keys
            logger.info("  [OK] [%s] %d active key(s) pooled", p_id.upper(), pool.total_keys)
        else:
            logger.warning("  [--] [%s] No keys configured in .env", p_id.upper())

    logger.info("-" * 65)
    logger.info("Ready to accept requests from Claude Desktop & Claude Code!")
    logger.info("=" * 65)

    yield

    logger.info("Shutting down LLM Router...")
    try:
        await router_engine.shutdown()
        logger.info("Provider connections closed gracefully.")
    except Exception as e:
        logger.debug("Error during shutdown: %s", e)


app = FastAPI(
    title="LLM Router",
    description="Unified resilient multi-provider router for Claude Desktop, Claude Code, and OpenAI tools",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web dashboard & tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(messages_router)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(health_router)
app.include_router(admin_router)


from fastapi.staticfiles import StaticFiles

FREELLM_DIST = Path(__file__).resolve().parent.parent / "freellmapi-main" / "client" / "dist"
if FREELLM_DIST.exists():
    app.mount("/portal", StaticFiles(directory=str(FREELLM_DIST), html=True), name="portal")


# Root Health Check endpoints
@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/hello", methods=["GET", "HEAD"])
@app.api_route("/ping", methods=["GET", "HEAD"])
async def root_health():
    return {"status": "healthy", "service": "llm-router"}


# Dashboard Home
@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    template_path = TEMPLATES_DIR / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>LLM Router Operational</h1><p>Dashboard template not found.</p>")
