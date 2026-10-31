"""Model discovery endpoint (/v1/models) for Claude Desktop, Claude Code, Cursor, and OpenAI clients."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from fastapi import APIRouter

from app.core.config import config

router = APIRouter(tags=["Models"])


@router.get("/v1/models")
@router.get("/models")
async def list_models() -> Dict[str, Any]:
    models: List[Dict[str, Any]] = []
    now = int(time.time())
    iso_now = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen_ids = set()

    def add_model(m_id: str, d_name: str, ctx: int, owner: str):
        if m_id in seen_ids:
            return
        seen_ids.add(m_id)
        models.append({
            "id": m_id,
            "object": "model",
            "type": "model",
            "created": now,
            "created_at": iso_now,
            "owned_by": owner,
            "permission": [],
            "root": m_id,
            "parent": None,
            "display_name": d_name,
            "context_window": ctx,
        })

    # 1. Primary Specialized Virtual Stacks
    primary_stacks = [
        ("hermes-agent", "Hermes Agent", 1048576, "llm-router"),
        ("pro-coding", "Pro Coding", 1048576, "llm-router"),
        ("deep-reasoning", "Deep Reasoning", 1048576, "llm-router"),
        ("frontier-intelligence", "Frontier Intelligence", 1048576, "llm-router"),
        ("massive-context", "Massive Context", 2097152, "llm-router"),
        ("ultra-fast-planning", "Ultra Fast Planning", 131072, "llm-router"),
        ("multimodal-vision", "Multimodal Vision", 1048576, "llm-router"),
        ("massive-quota-bulk", "Massive Quota Bulk", 131072, "llm-router"),
        ("mega-pool", "Mega Pool (10+ Providers)", 1048576, "llm-router"),
    ]
    for m_id, d_name, ctx, owner in primary_stacks:
        add_model(m_id, d_name, ctx, owner)

    # 2. All Models & Aliases from config.yaml (Claude, Gemini, OpenAI, DeepSeek, etc.)
    for model_id, model_cfg in config.models.items():
        if model_id in seen_ids:
            continue
        disp = getattr(model_cfg, "display_name", "") or model_id
        ctx = 128000
        owner = "llm-router"
        if model_id.startswith("claude-"):
            owner = "anthropic"
            ctx = 200000
        elif model_id.startswith("gemini-") or model_id.startswith("gemma-"):
            owner = "google"
            ctx = 1048576
        elif model_id.startswith("gpt-"):
            owner = "openai"
            ctx = 128000
        elif "deepseek" in model_id:
            owner = "deepseek"
            ctx = 64000
        add_model(model_id, disp, ctx, owner)

    # 3. Claude-Native Aliases for Router Stacks & Gemini Models
    # (Allows Claude Desktop & Claude Code CLI to discover and pick these directly)
    claude_aliases = [
        ("claude-gemini-3.6-flash", "Claude (Gemini 3.6 Flash)", 1048576, "anthropic"),
        ("claude-gemini-3-flash", "Claude (Gemini 3 Flash Preview)", 1048576, "anthropic"),
        ("claude-gemini-3.5-flash-lite", "Claude (Gemini 3.5 Flash Lite)", 1048576, "anthropic"),
        ("claude-gemma-4-31b", "Claude (Gemma 4 31B 14k RPD)", 131072, "anthropic"),
        ("claude-hermes-agent", "Claude (Hermes Agent Stack)", 1048576, "anthropic"),
        ("claude-pro-coding", "Claude (Pro Coding Stack)", 1048576, "anthropic"),
        ("claude-deep-reasoning", "Claude (Deep Reasoning Stack)", 1048576, "anthropic"),
        ("claude-mega-pool", "Claude (Mega Pool Bulk Stack)", 1048576, "anthropic"),
    ]
    for m_id, d_name, ctx, owner in claude_aliases:
        add_model(m_id, d_name, ctx, owner)

    return {
        "object": "list",
        "data": models,
        "has_more": False,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    }


@router.get("/v1/models/{model_id:path}")
@router.get("/models/{model_id:path}")
async def get_model(model_id: str) -> Dict[str, Any]:
    """Retrieve details for a specific model ID (OpenAI compatibility)."""
    model_list = await list_models()
    for m in model_list["data"]:
        if m["id"] == model_id:
            return m

    # Return valid generic model descriptor so clients never fail validation
    now = int(time.time())
    return {
        "id": model_id,
        "object": "model",
        "created": now,
        "owned_by": "llm-router",
        "permission": [],
        "root": model_id,
        "parent": None,
        "display_name": model_id,
        "context_window": 128000,
    }


# ----------------------------------------------------------------------
# Discovery & Compatibility Probes (Hermes, Ollama, LangChain clients)
# ----------------------------------------------------------------------
@router.get("/api/tags")
@router.get("/api/v1/models")
async def ollama_tags() -> Dict[str, Any]:
    model_list = await list_models()
    ollama_models = []
    for m in model_list["data"]:
        ollama_models.append({
            "name": m["id"],
            "model": m["id"],
            "modified_at": "2026-10-25T00:00:00Z",
            "size": 0,
            "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "router",
                "families": ["router"],
                "parameter_size": "70B",
                "quantization_level": "Q4_K_M",
            },
        })
    return {"models": ollama_models}


@router.get("/api/version")
@router.get("/version")
async def version_info() -> Dict[str, Any]:
    return {"version": "1.0.0"}


@router.get("/v1/props")
@router.get("/props")
async def props_info() -> Dict[str, Any]:
    return {"status": "ok", "service": "llm-router", "version": "1.0.0"}

