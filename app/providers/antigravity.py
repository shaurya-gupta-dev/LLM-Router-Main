"""Google Antigravity provider adapter for LLM Router.

Supports Antigravity-hosted models (gemini-3.8-flash, gemini-2.5-pro, etc.)
using local OAuth credentials, UI-configured tokens, or Google API keys.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.providers.gemini import GeminiProvider, _strip_thinking_preamble
from app.providers.base import (
    AuthException,
    ProviderException,
    RateLimitException,
    UnifiedChatResult,
)

logger = logging.getLogger(__name__)


def get_local_antigravity_token() -> Optional[str]:
    """Auto-discover OAuth access token from local state or credentials if unexpired."""
    # 1. Check state.json first (configured via UI / admin endpoint)
    try:
        from app.core.state_manager import state_manager
        ui_token = state_manager.get_antigravity_token()
        if ui_token and isinstance(ui_token, str) and ui_token.strip():
            return ui_token.strip()
    except Exception as e:
        logger.debug("Could not read antigravity token from state_manager: %s", e)

    # 2. Check local ~/.gemini/oauth_creds.json
    try:
        creds_path = Path.home() / ".gemini" / "oauth_creds.json"
        if creds_path.exists():
            with open(creds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            expiry_date = data.get("expiry_date")
            if expiry_date:
                now_ms = time.time() * 1000
                if now_ms > expiry_date:
                    logger.debug("Local antigravity token in ~/.gemini/oauth_creds.json is expired")
                    return None
            token = data.get("access_token")
            if token and isinstance(token, str):
                return token.strip()
    except Exception as e:
        logger.debug("Could not read local antigravity credentials: %s", e)

    return None


class AntigravityProvider(GeminiProvider):
    """Adapter for Antigravity-backed Gemini models."""

    def __init__(
        self,
        provider_id: str = "antigravity",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        super().__init__(provider_id, base_url)

    @classmethod
    def get_default_keys(cls) -> List[str]:
        """Check local credentials or state fallback."""
        token = get_local_antigravity_token()
        return [token] if token else []

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        api_key: str,
    ) -> UnifiedChatResult:
        clean_model = model.replace("models/", "")

        # Support Bearer tokens vs API keys
        is_oauth = api_key.startswith("ya29.") or api_key.startswith("Bearer ")
        headers = {"Content-Type": "application/json"}

        if is_oauth:
            token = api_key.replace("Bearer ", "")
            headers["Authorization"] = f"Bearer {token}"
            url = f"{self.base_url}/models/{clean_model}:generateContent"
        else:
            url = f"{self.base_url}/models/{clean_model}:generateContent?key={api_key}"

        sys_inst, contents = self._convert_messages_to_gemini(messages)
        payload: Dict[str, Any] = {"contents": contents}

        if sys_inst:
            payload["systemInstruction"] = sys_inst

        gen_cfg: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            gen_cfg["maxOutputTokens"] = max_tokens
        payload["generationConfig"] = gen_cfg

        gemini_tools = self._convert_tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        try:
            resp = await self.client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            raise ProviderException(f"Antigravity network error: {exc}", provider=self.provider_id) from exc

        if resp.status_code != 200:
            self._handle_gemini_error(resp.status_code, resp.text)

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return UnifiedChatResult(content="")

        first_cand = candidates[0]
        content_obj = first_cand.get("content", {})
        parts = content_obj.get("parts", [])

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for p in parts:
            if "text" in p:
                text_parts.append(p["text"])
            elif "functionCall" in p:
                fn = p["functionCall"]
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": json.dumps(fn.get("args", {})),
                    },
                })

        raw_content = "".join(text_parts)
        cleaned_content = _strip_thinking_preamble(raw_content)

        usage_md = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_md.get("promptTokenCount", 0),
            "completion_tokens": usage_md.get("candidatesTokenCount", 0),
            "total_tokens": usage_md.get("totalTokenCount", 0),
        }

        return UnifiedChatResult(
            content=cleaned_content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=first_cand.get("finishReason", "stop"),
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        api_key: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        clean_model = model.replace("models/", "")
        is_oauth = api_key.startswith("ya29.") or api_key.startswith("Bearer ")
        headers = {"Content-Type": "application/json"}

        if is_oauth:
            token = api_key.replace("Bearer ", "")
            headers["Authorization"] = f"Bearer {token}"
            url = f"{self.base_url}/models/{clean_model}:streamGenerateContent?alt=sse"
        else:
            url = f"{self.base_url}/models/{clean_model}:streamGenerateContent?alt=sse&key={api_key}"

        sys_inst, contents = self._convert_messages_to_gemini(messages)
        payload: Dict[str, Any] = {"contents": contents}

        if sys_inst:
            payload["systemInstruction"] = sys_inst

        gen_cfg: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            gen_cfg["maxOutputTokens"] = max_tokens
        payload["generationConfig"] = gen_cfg

        gemini_tools = self._convert_tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        req = self.client.build_request("POST", url, json=payload, headers=headers)
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.RequestError as exc:
            raise ProviderException(f"Network error connecting to Antigravity: {exc}", provider="antigravity")

        if resp.status_code != 200:
            await resp.aread()
            self._handle_gemini_error(resp.status_code, resp.text)

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue

            candidates = chunk.get("candidates") or []
            if not candidates:
                continue

            first_cand = candidates[0]
            content_obj = first_cand.get("content", {})
            parts = content_obj.get("parts") or []

            text_delta = ""
            for p in parts:
                if "text" in p:
                    text_delta += p["text"]

            if text_delta:
                yield {
                    "delta": text_delta,
                    "finish_reason": first_cand.get("finishReason"),
                }
