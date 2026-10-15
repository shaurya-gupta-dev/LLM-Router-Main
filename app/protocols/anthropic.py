"""Anthropic Messages API protocol schemas and SSE event generators."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, List[Dict[str, Any]]]


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: int = Field(default=4096, gt=0)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    thinking: Optional[Dict[str, Any]] = None


def create_anthropic_response(
    model: str,
    content_blocks: List[Dict[str, Any]],
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standard Anthropic non-streaming message response dictionary."""
    return {
        "id": msg_id or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


async def format_anthropic_sse(
    event_type: str, data: Dict[str, Any]
) -> str:
    """Format a single Anthropic Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def generate_anthropic_stream_from_chunks(
    model: str,
    chunk_generator: AsyncGenerator[Dict[str, Any], None],
    initial_input_tokens: int = 0,
) -> AsyncGenerator[str, None]:
    """
    Consumes unified delta chunks from any provider and yields
    conforming Anthropic Server-Sent Events (SSE).
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': initial_input_tokens, 'output_tokens': 1}}})}\n\n"

    current_block_index = 0
    text_block_open = False
    active_tool_call_index: Optional[int] = None
    active_tool_calls: Dict[int, Dict[str, Any]] = {}
    output_tokens = 0
    final_stop_reason = "end_turn"

    async for chunk in chunk_generator:
        # 1. Text Delta
        text = chunk.get("text")
        if text:
            output_tokens += 1
            if not text_block_open:
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_block_open = True

            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_block_index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

        # 2. Tool Calls
        tool_call = chunk.get("tool_call")
        if tool_call:
            # If text block was open, close it first
            if text_block_open:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
                text_block_open = False
                current_block_index += 1

            tc_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            tc_name = tool_call.get("name", "")
            tc_args_delta = tool_call.get("arguments_delta", "")

            tc_idx = tool_call.get("index", 0)
            if tc_idx not in active_tool_calls:
                # Start new tool_use block
                active_tool_calls[tc_idx] = {"id": tc_id, "name": tc_name, "block_index": current_block_index}
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_block_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': tc_name, 'input': {}}})}\n\n"
                current_block_index += 1

            target_block_idx = active_tool_calls[tc_idx]["block_index"]
            if tc_args_delta:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': target_block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': tc_args_delta}})}\n\n"

        # 3. Stop Reason & Usage from chunk
        if chunk.get("stop_reason"):
            sr = chunk["stop_reason"]
            if sr in ("tool_calls", "function_call", "tool_use"):
                final_stop_reason = "tool_use"
            elif sr in ("stop", "end_turn"):
                final_stop_reason = "end_turn"
            elif sr in ("length", "max_tokens"):
                final_stop_reason = "max_tokens"

        if chunk.get("output_tokens"):
            output_tokens = max(output_tokens, chunk["output_tokens"])

    # Close any open tool blocks
    for tc_info in active_tool_calls.values():
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tc_info['block_index']})}\n\n"

    # Close text block if still open
    if text_block_open:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"

    # Message Delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

    # Message Stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
