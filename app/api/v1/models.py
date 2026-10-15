"""Model discovery endpoint (/v1/models) for Claude Desktop, Cursor, and clients."""

from __future__ import annotations

import time
from typing import Any, Dict, List
from fastapi import APIRouter

from app.core.config import config

router = APIRouter(tags=["Models"])


@router.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    models: List[Dict[str, Any]] = []
    now = int(time.time())

    # 1. The 3 Primary Virtual Task-Centric Stacks
    core_virtual_stacks = [
        ("pro-coding-reasoning", "Pro Coding & Reasoning", 128000),
        ("fast-planning-light", "Fast Planning & Light Logic", 128000),
        ("general-massive-quota", "General Tasks & Massive Quota", 128000),
    ]
    for m_id, d_name, ctx in core_virtual_stacks:
        models.append({
            "id": m_id,
            "object": "model",
            "created": now,
            "owned_by": "llm-router",
            "permission": [],
            "root": m_id,
            "parent": None,
            "display_name": d_name,
            "context_window": ctx,
        })

    # 2. Seamless Claude Desktop & Anthropic Client Aliases (Exactly 3 clean 1-to-1 mapped options)
    claude_aliases = [
        ("claude-3-7-sonnet-latest", "Pro Coding & Reasoning (Claude 3.7 Sonnet)", 200000),
        ("claude-3-5-haiku-20241022", "Fast Planning & Light Logic (Claude 3.5 Haiku)", 200000),
        ("claude-3-5-sonnet-20241022", "General Tasks & Massive Quota (Claude 3.5 Sonnet)", 200000),
    ]
    for m_id, d_name, ctx in claude_aliases:
        models.append({
            "id": m_id,
            "object": "model",
            "created": now,
            "owned_by": "anthropic",
            "permission": [],
            "root": m_id,
            "parent": None,
            "display_name": d_name,
            "context_window": ctx,
        })

    return {
        "object": "list",
        "data": models,
    }
