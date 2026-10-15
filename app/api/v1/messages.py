"""Anthropic Messages API Endpoint (/v1/messages) for Claude Desktop & Claude Code."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import config
from app.core.router import router_engine
from app.protocols.anthropic import (
    AnthropicMessagesRequest,
    create_anthropic_response,
    generate_anthropic_stream_from_chunks,
)
from app.protocols.tool_translator import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    openai_tool_calls_to_anthropic_blocks,
)
from app.providers.base import ProviderException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Anthropic"])


def verify_auth(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    master = config.settings.master_key.strip()
    if not master:
        return  # No master key required

    token = x_api_key
    if not token and authorization:
        token = authorization.removeprefix("Bearer ").strip()

    if not token or token != master:
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key or missing x-api-key / Authorization header",
                },
            },
        )


@router.post("/v1/messages")
async def handle_anthropic_messages(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    verify_auth(x_api_key, authorization)

    try:
        raw_body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        req = AnthropicMessagesRequest(**raw_body)
    except Exception as e:
        logger.error("Validation error on /v1/messages: %s", e)
        raise HTTPException(
            status_code=400,
            detail={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}},
        )

    # 1. Translate Anthropic messages & tools into unified standard representation
    raw_messages = [m.model_dump() for m in req.messages]
    standard_messages = anthropic_messages_to_openai(raw_messages, system=req.system)
    standard_tools = anthropic_tools_to_openai(req.tools)

    # 2. Execute via Resilient Router
    if req.stream:
        try:
            chunk_gen, resolved_prov, resolved_model, metadata = await router_engine.execute_stream(
                requested_model=req.model,
                messages=standard_messages,
                tools=standard_tools,
                max_tokens=req.max_tokens,
                temperature=req.temperature or 0.7,
            )
            sse_generator = generate_anthropic_stream_from_chunks(
                model=req.model,
                chunk_generator=chunk_gen,
                initial_input_tokens=len(str(standard_messages)) // 4,
            )
            return StreamingResponse(
                sse_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "anthropic-version": "2023-06-01",
                    "X-Routed-Provider": resolved_prov,
                    "X-Routed-Model": resolved_model,
                    "X-Router-Tier": metadata.get("tier", "UNKNOWN"),
                    "X-Router-Classifier": f"{metadata.get('classification_method')} ({metadata.get('classification_ms')}ms)",
                    "X-Router-Attempts": str(len(metadata.get("candidates_attempted", []))),
                },
            )
        except ProviderException as pe:
            logger.error("Provider error during streaming /v1/messages: %s", pe.message)
            raise HTTPException(
                status_code=pe.status_code,
                detail={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": pe.message,
                        "diagnostics": pe.diagnostics,
                    },
                },
            )

    else:
        # Non-streaming
        try:
            result, resolved_prov, resolved_model, metadata = await router_engine.execute_chat(
                requested_model=req.model,
                messages=standard_messages,
                tools=standard_tools,
                max_tokens=req.max_tokens,
                temperature=req.temperature or 0.7,
            )

            # Build Anthropic content blocks
            content_blocks: list[dict[str, Any]] = []
            if result.content or not result.tool_calls:
                content_blocks.append({"type": "text", "text": result.content or ""})

            if result.tool_calls:
                content_blocks.extend(openai_tool_calls_to_anthropic_blocks(result.tool_calls))

            stop_reason = "end_turn"
            if result.stop_reason == "tool_calls" or result.tool_calls:
                stop_reason = "tool_use"
            elif result.stop_reason in ("length", "max_tokens"):
                stop_reason = "max_tokens"

            response_payload = create_anthropic_response(
                model=req.model,
                content_blocks=content_blocks,
                stop_reason=stop_reason,
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            )

            return JSONResponse(
                content=response_payload,
                headers={
                    "anthropic-version": "2023-06-01",
                    "X-Routed-Provider": resolved_prov,
                    "X-Routed-Model": resolved_model,
                    "X-Router-Tier": metadata.get("tier", "UNKNOWN"),
                    "X-Router-Classifier": f"{metadata.get('classification_method')} ({metadata.get('classification_ms')}ms)",
                    "X-Router-Attempts": str(len(metadata.get("candidates_attempted", []))),
                },
            )

        except ProviderException as pe:
            logger.error("Provider error during /v1/messages: %s", pe.message)
            raise HTTPException(
                status_code=pe.status_code,
                detail={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": pe.message,
                        "diagnostics": pe.diagnostics,
                    },
                },
            )
