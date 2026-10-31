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
        pool = key_manager.get_pool(p_id)
        has_keys = pool is not None and pool.total_keys > 0

        existing_ids = set()
        models_list = []

        # 1. Statically configured models from config.yaml
        for m in p_def.models:
            existing_ids.add(m.id)
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
                "is_dynamic": False,
            })

        # 2. Dynamically discovered live models from upstream provider
        disc_models = state_manager.get_discovered_models(p_id)
        for dm in disc_models:
            dm_id = dm.get("id")
            if dm_id and dm_id not in existing_ids:
                existing_ids.add(dm_id)
                key = f"{p_id}:{dm_id}"
                health = health_cache.get(key, {})
                models_list.append({
                    "id": dm_id,
                    "name": dm.get("name") or dm_id,
                    "context_window": dm.get("context_window", 128000),
                    "supports_tools": dm.get("supports_tools", True),
                    "supports_vision": dm.get("supports_vision", False),
                    "supports_thinking": dm.get("supports_thinking", False),
                    "disabled": disabled_map.get(key, False),
                    "health_status": health.get("status", "unknown"),
                    "health_latency_ms": health.get("latency_ms"),
                    "last_checked": health.get("last_checked"),
                    "is_dynamic": True,
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
    """
    Live-query upstream provider /models endpoints to discover newly offered models,
    then probe operational health and update the dynamic state cache.
    """
    from app.core.router import router_engine

    health_results: Dict[str, Any] = {}
    discovery_summary: Dict[str, int] = {}
    total_discovered_new = 0

    # -------------------------------------------------------------------------
    # PHASE 1: Live Dynamic Model Discovery from Upstream Provider APIs
    # -------------------------------------------------------------------------
    async def discover_provider_models(p_id: str, provider):
        nonlocal total_discovered_new
        key_state = key_manager.get_key(p_id)
        if not key_state or not key_state.is_available:
            return

        try:
            live_models = await asyncio.wait_for(provider.list_models(key_state.key), timeout=12.0)
            if live_models:
                configured_ids = {m.id for m in config.providers[p_id].models} if p_id in config.providers else set()
                new_models = [m for m in live_models if m.get("id") not in configured_ids]
                state_manager.set_discovered_models(p_id, live_models)
                discovery_summary[p_id] = len(live_models)
                total_discovered_new += len(new_models)
                logger.info(
                    "Dynamic discovery for '%s': %d total models online (%d new)",
                    p_id,
                    len(live_models),
                    len(new_models),
                )
        except Exception as e:
            logger.debug("Live discovery skipped for '%s': %s", p_id, e)

    disc_tasks = []
    for p_id in config.providers.keys():
        prov = router_engine.providers.get(p_id)
        if prov:
            disc_tasks.append(discover_provider_models(p_id, prov))

    if disc_tasks:
        await asyncio.gather(*disc_tasks, return_exceptions=True)

    # -------------------------------------------------------------------------
    # PHASE 2: Live Model Health Probing
    # -------------------------------------------------------------------------
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
            await asyncio.wait_for(
                provider.chat_completion(
                    model=model_id,
                    messages=[{"role": "user", "content": "Hi"}],
                    tools=None,
                    max_tokens=5,
                    temperature=0.0,
                    api_key=key_state.key,
                ),
                timeout=12.0,
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
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                status = "rate_limited"
            elif "subscription" in error_msg:
                status = "subscription_required"
            elif "404" in error_msg or "not found" in error_msg or "deprecated" in error_msg:
                status = "unavailable"
            else:
                status = "error"

            health_results[key] = {
                "status": status,
                "latency_ms": None,
                "last_checked": time.time(),
                "error": str(e)[:120],
            }

    for p_id, p_def in config.providers.items():
        provider = router_engine.providers.get(p_id)
        if not provider:
            continue

        # Statically configured models
        models_to_probe = [m.id for m in p_def.models]

        # Plus discovered models (cap at 6 dynamic models per provider to avoid quota exhaustion)
        disc = state_manager.get_discovered_models(p_id)
        for dm in disc[:6]:
            d_id = dm.get("id")
            if d_id and d_id not in models_to_probe:
                models_to_probe.append(d_id)

        for m_id in models_to_probe:
            tasks.append(probe_model(p_id, m_id, provider))

    # Run probes concurrently with bounded concurrency
    sem = asyncio.Semaphore(6)

    async def bounded_probe(coro):
        async with sem:
            await coro

    await asyncio.gather(*[bounded_probe(t) for t in tasks], return_exceptions=True)

    state_manager.update_health_cache(health_results)

    healthy = sum(1 for v in health_results.values() if v["status"] == "healthy")
    total = len(health_results)

    return {
        "status": "ok",
        "total_models": total,
        "healthy": healthy,
        "unhealthy": total - healthy,
        "discovered_new": total_discovered_new,
        "discovery_summary": discovery_summary,
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


# ----------------------------------------------------------------------
# Request History & Audit Endpoints
# ----------------------------------------------------------------------
@router.get("/history")
async def get_history_endpoint(
    limit: int = 100,
    status: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve filtered request history log and fulfillment metrics."""
    from app.core.history_manager import history_manager
    return history_manager.get_history(limit=limit, status=status, provider=provider)


@router.delete("/history")
async def clear_history_endpoint() -> Dict[str, Any]:
    """Clear all historical request records."""
    from app.core.history_manager import history_manager
    history_manager.clear_history()
    return {"status": "ok", "message": "History cleared"}


# ----------------------------------------------------------------------
# Real-Time Usage & Limits Endpoints
# ----------------------------------------------------------------------
@router.get("/usage")
async def get_comprehensive_usage_endpoint() -> Dict[str, Any]:
    """Get live RPM/RPD/TPM usage, quota limits, and pool status for all providers and models."""
    return rate_tracker.get_comprehensive_usage()


@router.get("/usage/provider/{provider_id}")
async def get_provider_usage_endpoint(provider_id: str) -> Dict[str, Any]:
    """Get detailed usage and model metrics for a specific provider."""
    data = rate_tracker.get_comprehensive_usage()
    for p in data.get("providers", []):
        if p["id"] == provider_id:
            return p
    raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")


@router.post("/usage/reset")
async def reset_usage_endpoint() -> Dict[str, Any]:
    """Reset live usage counters."""
    rate_tracker.reset_metrics()
    return {"status": "ok", "message": "Usage counters reset"}

