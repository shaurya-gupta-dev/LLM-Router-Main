"""OpenAI Chat Completions API Endpoint (/v1/chat/completions)."""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import config
from app.core.router import router_engine
from app.protocols.openai import (
    ChatCompletionRequest,
    create_openai_response,
    generate_openai_stream_from_chunks,
)
from app.providers.base import ProviderException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OpenAI"])


def verify_auth(
    authorization: Optional[str] = Header(None, alias="authorization"),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    master = config.settings.master_key.strip()
    if not master:
        return

    token = (x_api_key or "").strip()
    if not token and authorization:
        auth_val = authorization.strip()
        if auth_val.lower().startswith("bearer "):
            token = auth_val[7:].strip()
        else:
            token = auth_val

    if not token or token != master:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key or missing Authorization / x-api-key header",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


@router.post("/v1/chat/completions")
async def handle_chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None, alias="authorization"),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    verify_auth(authorization, x_api_key)

    try:
        raw_body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        req = ChatCompletionRequest(**raw_body)
    except Exception as e:
        logger.error("Validation error on /v1/chat/completions: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    if req.stream:
        try:
            chunk_gen, resolved_prov, resolved_model, metadata = await router_engine.execute_stream(
                requested_model=req.model,
                messages=messages,
                tools=req.tools,
                max_tokens=req.max_tokens,
                temperature=req.temperature or 0.7,
            )
            sse_generator = generate_openai_stream_from_chunks(
                model=req.model,
                chunk_generator=chunk_gen,
            )
            return StreamingResponse(
                sse_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Routed-Provider": resolved_prov,
                    "X-Routed-Model": resolved_model,
                    "X-Router-Tier": metadata.get("tier", "UNKNOWN"),
                    "X-Router-Classifier": f"{metadata.get('classification_method')} ({metadata.get('classification_ms')}ms)",
                    "X-Router-Attempts": str(len(metadata.get("candidates_attempted", []))),
                },
            )
        except ProviderException as pe:
            raise HTTPException(
                status_code=pe.status_code,
                detail={
                    "error": {
                        "message": pe.message,
                        "type": "provider_error",
                        "diagnostics": pe.diagnostics,
                    }
                },
            )
    else:
        try:
            result, resolved_prov, resolved_model, metadata = await router_engine.execute_chat(
                requested_model=req.model,
                messages=messages,
                tools=req.tools,
                max_tokens=req.max_tokens,
                temperature=req.temperature or 0.7,
            )
            payload = create_openai_response(
                model=req.model,
                content=result.content,
                tool_calls=result.tool_calls,
                stop_reason=result.stop_reason,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            return JSONResponse(
                content=payload,
                headers={
                    "X-Routed-Provider": resolved_prov,
                    "X-Routed-Model": resolved_model,
                    "X-Router-Tier": metadata.get("tier", "UNKNOWN"),
                    "X-Router-Classifier": f"{metadata.get('classification_method')} ({metadata.get('classification_ms')}ms)",
                    "X-Router-Attempts": str(len(metadata.get("candidates_attempted", []))),
                },
            )
        except ProviderException as pe:
            raise HTTPException(
                status_code=pe.status_code,
                detail={
                    "error": {
                        "message": pe.message,
                        "type": "provider_error",
                        "diagnostics": pe.diagnostics,
                    }
                },
            )
