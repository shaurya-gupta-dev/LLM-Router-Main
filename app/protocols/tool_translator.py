"""Bidirectional tool call and schema translator for Anthropic, OpenAI, and Gemini formats."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Anthropic tool definitions to OpenAI tool definitions."""
    if not tools:
        return None

    openai_tools = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # If already in OpenAI format:
        if "type" in t and "function" in t:
            openai_tools.append(t)
            continue

        fn_def = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        }
        openai_tools.append({
            "type": "function",
            "function": fn_def,
        })
    return openai_tools


def openai_tools_to_anthropic(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI tool definitions to Anthropic tool definitions."""
    if not tools:
        return None

    anthropic_tools = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        elif "input_schema" in t:
            anthropic_tools.append(t)
    return anthropic_tools


def anthropic_tools_to_gemini(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Anthropic tools to Gemini function declarations format."""
    if not tools:
        return None

    declarations = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema") or (t.get("function", {}).get("parameters")) or {}

        # Normalize schema for Gemini: properties type strings need to be uppercase in strict mode,
        # but modern v1beta accepts standard JSON Schema objects.
        declarations.append({
            "name": name,
            "description": desc,
            "parameters": schema,
        })

    return [{"function_declarations": declarations}] if declarations else None


def anthropic_messages_to_openai(
    messages: List[Dict[str, Any]], system: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Convert Anthropic messages (with system prompt, content blocks, tool_use, tool_result) to OpenAI format."""
    openai_msgs: List[Dict[str, Any]] = []

    # 1. Handle system prompt
    if system:
        sys_str = ""
        if isinstance(system, str):
            sys_str = system
        elif isinstance(system, list):
            # Array of text blocks
            parts = []
            for item in system:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            sys_str = "\n\n".join(parts)
        if sys_str:
            openai_msgs.append({"role": "system", "content": sys_str})

    # 2. Convert messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts = []
            tool_calls = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))

                elif block_type == "tool_use":
                    tool_id = block.get("id", f"call_{len(tool_calls)}")
                    name = block.get("name", "")
                    inp = block.get("input", {})
                    args_str = json.dumps(inp) if isinstance(inp, dict) else str(inp)
                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str,
                        },
                    })

                elif block_type == "tool_result":
                    # Tool result corresponds to an OpenAI role: "tool"
                    t_id = block.get("tool_use_id", "")
                    res_content = block.get("content", "")
                    if isinstance(res_content, list):
                        res_parts = [
                            b.get("text", "") for b in res_content if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        res_content = "\n".join(res_parts)
                    elif not isinstance(res_content, str):
                        res_content = json.dumps(res_content)

                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": t_id,
                        "content": res_content,
                    })

            # If there was an assistant text or tool calls
            if role == "assistant":
                asst_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    asst_msg["tool_calls"] = tool_calls
                openai_msgs.append(asst_msg)
            elif role == "user" and text_parts:
                openai_msgs.append({
                    "role": "user",
                    "content": "\n".join(text_parts),
                })

    return openai_msgs


def openai_tool_calls_to_anthropic_blocks(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool_calls array to Anthropic tool_use content blocks."""
    blocks = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            parsed_args = {}

        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"call_{len(blocks)}"),
            "name": fn.get("name", ""),
            "input": parsed_args,
        })
    return blocks
