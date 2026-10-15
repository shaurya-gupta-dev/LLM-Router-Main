"""Health and monitoring endpoints for LLM Router."""

from __future__ import annotations

import time
from typing import Any, Dict
from fastapi import APIRouter

from app.core.config import config
from app.core.key_pool import key_manager
from app.core.rate_tracker import rate_tracker
from app.core.router import router_engine

router = APIRouter(prefix="/api", tags=["Monitoring"])


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Expose non-sensitive config to dashboard (master key shown so users can copy it)."""
    return {
        "master_key": config.settings.master_key,
        "port": config.settings.router_port,
        "host": config.settings.router_host,
        "version": "1.0.0",
        "providers_configured": len(config.providers),
        "models_configured": len(config.models),
    }


@router.api_route("/ping", methods=["GET", "HEAD"])
@router.api_route("/health", methods=["GET", "HEAD"])
@router.api_route("/hello", methods=["GET", "HEAD"])
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "providers_configured": len(config.providers),
        "models_configured": len(config.models),
    }


from app.core.rate_budget import rate_budget


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    stats = rate_tracker.get_overview()
    stats["key_pools"] = key_manager.get_status()
    stats["rate_budget"] = rate_budget.get_status()

    # Calculate global pooled capacities across all registered providers
    total_rpd_cap = sum(p.free_tier.rpd for p in config.providers.values())
    total_rpm_cap = sum(p.free_tier.rpm for p in config.providers.values())
    # Monthly combined token pool across all free tiers (~1.25 Billion tokens)
    total_tokens_cap = 1_250_000_000

    tokens_used = stats.get("total_tokens", 0)
    reqs_used = stats.get("total_requests", 0)

    stats["global_limits"] = {
        "total_tokens_capacity": total_tokens_cap,
        "total_tokens_used": tokens_used,
        "tokens_used_pct": round((tokens_used / total_tokens_cap) * 100, 4) if total_tokens_cap else 0.0,
        "total_rpd_capacity": total_rpd_cap,
        "total_requests_used": reqs_used,
        "rpd_used_pct": round((reqs_used / total_rpd_cap) * 100, 2) if total_rpd_cap else 0.0,
        "total_rpm_capacity": total_rpm_cap,
        "current_rpm": stats.get("current_rpm", 0),
    }
    return stats


@router.get("/chains")
async def get_chains() -> Dict[str, Any]:
    chains: Dict[str, Any] = {}
    for alias_id, alias_cfg in config.models.items():
        chains[alias_id] = {
            "display_name": alias_cfg.display_name,
            "fallbacks": [
                {
                    "provider": fb.provider,
                    "model": fb.model,
                    "provider_name": config.providers[fb.provider].name if fb.provider in config.providers else fb.provider,
                }
                for fb in alias_cfg.fallbacks
            ],
        }
    return chains


@router.post("/reload-keys")
async def reload_keys() -> Dict[str, Any]:
    config.load()
    router_engine.reload_keys()
    kp = key_manager.get_status()
    active_providers = [p for p, v in kp.items() if v.get("total_keys", 0) > 0]
    return {
        "status": "reloaded",
        "key_pools": kp,
        "active_count": len(active_providers),
        "active_providers": active_providers,
    }
