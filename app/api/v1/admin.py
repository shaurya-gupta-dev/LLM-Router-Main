"""Admin API endpoints for LLM Router control panel.

Provides endpoints for:
- Model discovery, enable/disable toggles
- Live model health checking (recheck)
- Routing strategy management (presets, custom priority)
- State inspection
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import config
from app.core.key_pool import key_manager
from app.core.rate_budget import rate_budget
from app.core.rate_tracker import rate_tracker
from app.core.state_manager import state_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# -------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------
class ToggleModelRequest(BaseModel):
    provider: str
    model: str
    disabled: bool


class BulkToggleRequest(BaseModel):
    toggles: Dict[str, bool]  # "provider:model" -> bool


class SetStrategyRequest(BaseModel):
    strategy: str  # "default", "max_speed", "best_quality", "max_quota", "direct", "custom"


class CustomPriorityRequest(BaseModel):
    priority: List[List[str]]  # [[provider, model], ...]


class SavePresetRequest(BaseModel):
    name: str
    display_name: str
    description: str
    priority: List[List[str]]


# -------------------------------------------------------------------
# Model Discovery & Control
# -------------------------------------------------------------------
@router.get("/models")
async def list_all_models() -> Dict[str, Any]:
    """List all detected models grouped by provider, with enable/disable status."""
    disabled_map = state_manager.get_disabled_models()
    health_cache, last_recheck = state_manager.get_health_cache()

    providers_data = {}
    for p_id, p_def in config.providers.items():
        # Check if provider has active keys
        pool = key_manager.get_pool(p_id)
        has_keys = pool is not None and pool.total_keys > 0

        models_list = []
        for m in p_def.models:
            key = f"{p_id}:{m.id}"
            health = health_cache.get(key, {})
            models_list.append({
                "id": m.id,
                "name": m.name,
                "context_window": m.context_window,
                "supports_tools": m.supports_tools,
                "supports_vision": m.supports_vision,
                "supports_thinking": m.supports_thinking,
                "disabled": disabled_map.get(key, False),
                "health_status": health.get("status", "unknown"),
                "health_latency_ms": health.get("latency_ms"),
                "last_checked": health.get("last_checked"),
            })

        providers_data[p_id] = {
            "name": p_def.name,
            "type": p_def.type,
            "base_url": p_def.base_url,
            "docs_url": p_def.docs_url,
            "has_keys": has_keys,
            "key_count": pool.total_keys if pool else 0,
            "free_tier": {
                "rpm": p_def.free_tier.rpm,
                "rpd": p_def.free_tier.rpd,
                "tpm": p_def.free_tier.tpm,
            },
            "models": models_list,
        }

    return {
        "providers": providers_data,
        "last_recheck": last_recheck,
        "active_strategy": state_manager.get_active_strategy(),
    }


@router.post("/models/toggle")
async def toggle_model(req: ToggleModelRequest) -> Dict[str, Any]:
    """Enable or disable a specific model."""
    state_manager.toggle_model(req.provider, req.model, req.disabled)
    return {
        "status": "ok",
        "provider": req.provider,
        "model": req.model,
        "disabled": req.disabled,
    }


@router.post("/models/bulk-toggle")
async def bulk_toggle(req: BulkToggleRequest) -> Dict[str, Any]:
    """Bulk enable/disable models."""
    state_manager.set_bulk_disabled(req.toggles)
    return {"status": "ok", "count": len(req.toggles)}


@router.post("/models/recheck")
async def recheck_models() -> Dict[str, Any]:
    """Live-probe all providers to verify which models are truly available."""
    from app.core.router import router_engine

    health_results: Dict[str, Any] = {}
    tasks = []

    async def probe_model(p_id: str, model_id: str, provider):
        key_state = key_manager.get_key(p_id)
        key = f"{p_id}:{model_id}"
        if not key_state:
            health_results[key] = {
                "status": "no_key",
                "latency_ms": None,
                "last_checked": time.time(),
            }
            return

        start = time.time()
        try:
            result = await asyncio.wait_for(
                provider.chat_completion(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hi"}],
                    tools=None,
                    max_tokens=5,
                    temperature=0.0,
                    api_key=key_state.key,
                ),
                timeout=15.0,
            )
            latency = (time.time() - start) * 1000
            health_results[key] = {
                "status": "healthy",
                "latency_ms": round(latency, 1),
                "last_checked": time.time(),
            }
        except asyncio.TimeoutError:
            health_results[key] = {
                "status": "timeout",
                "latency_ms": None,
                "last_checked": time.time(),
            }
        except Exception as e:
            error_msg = str(e)[:120]
            status = "rate_limited" if "429" in error_msg or "rate" in error_msg.lower() else "error"
            health_results[key] = {
                "status": status,
                "latency_ms": None,
                "last_checked": time.time(),
                "error": error_msg,
            }

    for p_id, p_def in config.providers.items():
        provider = router_engine.providers.get(p_id)
        if not provider:
            continue
        for m in p_def.models:
            tasks.append(probe_model(p_id, m.id, provider))

    # Run all probes concurrently with a semaphore to avoid flood
    sem = asyncio.Semaphore(5)

    async def bounded_probe(coro):
        async with sem:
            await coro

    await asyncio.gather(*[bounded_probe(t) for t in tasks], return_exceptions=True)

    state_manager.update_health_cache(health_results)

    # Count stats
    healthy = sum(1 for v in health_results.values() if v["status"] == "healthy")
    total = len(health_results)

    return {
        "status": "ok",
        "total_models": total,
        "healthy": healthy,
        "unhealthy": total - healthy,
        "results": health_results,
        "timestamp": time.time(),
    }


# -------------------------------------------------------------------
# Routing Strategy
# -------------------------------------------------------------------
@router.get("/strategy")
async def get_strategy() -> Dict[str, Any]:
    """Get current routing strategy and all available presets."""
    return {
        "active": state_manager.get_active_strategy(),
        "presets": state_manager.get_all_presets(),
        "custom_priority": state_manager.get_custom_priority(),
    }


@router.post("/strategy/set")
async def set_strategy(req: SetStrategyRequest) -> Dict[str, Any]:
    """Switch the active routing strategy."""
    valid = ["default", "max_speed", "best_quality", "max_quota", "direct", "custom"]
    presets = state_manager.get_all_presets()
    valid.extend(presets.keys())

    if req.strategy not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'. Valid: {valid}")

    state_manager.set_active_strategy(req.strategy)
    return {"status": "ok", "active_strategy": req.strategy}


@router.post("/strategy/custom-priority")
async def set_custom_priority(req: CustomPriorityRequest) -> Dict[str, Any]:
    """Set a custom priority ordering for the 'custom' strategy."""
    state_manager.set_custom_priority(req.priority)
    return {"status": "ok", "priority": req.priority}


@router.post("/strategy/save-preset")
async def save_preset(req: SavePresetRequest) -> Dict[str, Any]:
    """Save a named routing preset."""
    state_manager.save_preset(req.name, {
        "name": req.display_name,
        "description": req.description,
        "priority": [list(p) for p in req.priority],
    })
    return {"status": "ok", "preset": req.name}


@router.delete("/strategy/preset/{name}")
async def delete_preset(name: str) -> Dict[str, Any]:
    """Delete a custom routing preset."""
    if state_manager.delete_preset(name):
        return {"status": "ok", "deleted": name}
    raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")


# -------------------------------------------------------------------
# State
# -------------------------------------------------------------------
@router.get("/state")
async def get_state() -> Dict[str, Any]:
    """Get full state dump for debugging."""
    return state_manager.get_full_state()


@router.post("/state/reload")
async def reload_state() -> Dict[str, Any]:
    """Reload state.json from disk."""
    state_manager.reload()
    return {"status": "reloaded"}


# -------------------------------------------------------------------
# Antigravity UI Authentication
# -------------------------------------------------------------------
class SetAntigravityAuthRequest(BaseModel):
    token: str


@router.get("/auth/antigravity")
async def get_antigravity_auth() -> Dict[str, Any]:
    """Check Antigravity auth status."""
    token = state_manager.get_antigravity_token()
    return {
        "configured": bool(token),
        "token_prefix": (token[:10] + "...") if token else None,
    }


@router.post("/auth/antigravity")
async def set_antigravity_auth(req: SetAntigravityAuthRequest) -> Dict[str, Any]:
    """Save an explicit Antigravity OAuth Token or Google API key from the UI."""
    token = req.token.strip()
    state_manager.set_antigravity_token(token)

    # Update key pool dynamically
    pool = key_manager.get_pool("antigravity")
    if pool:
        pool.update_keys([token] if token else [])
    else:
        key_manager.register_provider("antigravity", [token] if token else [])

    p_def = config.providers.get("antigravity")
    if p_def:
        rate_budget.register_provider(
            "antigravity",
            rpm=p_def.free_tier.rpm,
            rpd=p_def.free_tier.rpd,
        )
    logger.info("Antigravity token updated via UI: configured=%s", bool(token))
    return {"status": "ok", "has_token": bool(token)}


@router.post("/auth/antigravity/auto-detect")
async def auto_detect_antigravity_token() -> Dict[str, Any]:
    """Auto-scan local environment and credentials for Antigravity access token or Google key."""
    from app.providers.antigravity import get_local_antigravity_token
    token = get_local_antigravity_token()
    if token:
        state_manager.set_antigravity_token(token)
        pool = key_manager.get_pool("antigravity")
        if pool:
            pool.update_keys([token])
        else:
            key_manager.register_provider("antigravity", [token])
        return {
            "status": "ok",
            "found": True,
            "token_prefix": token[:10] + "...",
            "message": "Local Antigravity token auto-detected and activated successfully!"
        }
    return {
        "status": "not_found",
        "found": False,
        "message": "No unexpired local Antigravity token found. You can paste your free Google AI Studio key (aistudio.google.com/apikey) or sign in."
    }
