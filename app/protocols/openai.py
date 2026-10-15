"""OpenAI Chat Completions protocol schemas and SSE event generators."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)


def create_openai_response(
    model: str,
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    stop_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    """Build a standard OpenAI chat completion response dictionary."""
    message: Dict[str, Any] = {"role": "assistant", "content": content if content is not None else ""}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def generate_openai_stream_from_chunks(
    model: str,
    chunk_generator: AsyncGenerator[Dict[str, Any], None],
) -> AsyncGenerator[str, None]:
    """Yields conforming OpenAI Server-Sent Events (data: {...}) and data: [DONE]."""
    req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Initial chunk with role
    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    async for chunk in chunk_generator:
        delta: Dict[str, Any] = {}
        if chunk.get("text"):
            delta["content"] = chunk["text"]

        if chunk.get("tool_call"):
            tc = chunk["tool_call"]
            delta["tool_calls"] = [
                {
                    "index": tc.get("index", 0),
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments_delta", ""),
                    },
                }
            ]

        finish_reason = chunk.get("stop_reason")
        if delta or finish_reason:
            payload = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"
