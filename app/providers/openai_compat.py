"""Universal OpenAI-compatible provider adapter (Groq, GitHub, Cerebras, OpenRouter, etc.)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.providers.base import (
    AuthException,
    BaseProvider,
    ProviderException,
    RateLimitException,
    UnifiedChatResult,
)

logger = logging.getLogger(__name__)

# Strip <think>...</think> blocks that DeepSeek-R1 / QwQ / similar embed in content
_THINK_TAG_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL | re.IGNORECASE)


class OpenAICompatProvider(BaseProvider):
    def __init__(self, provider_id: str, base_url: str):
        super().__init__(provider_id, base_url)
        # Shared async HTTP client with HTTP/2 (if h2 installed) and keepalive
        use_http2 = False
        try:
            import h2  # noqa: F401
            use_http2 = True
        except ImportError:
            pass
        connect_timeout = 8.0 if provider_id == "apinex" else 15.0
        read_timeout = 35.0 if provider_id == "apinex" else 90.0
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=15.0, pool=30.0),
            http2=use_http2,
            follow_redirects=True,
        )

    def _resolve_url(self, api_key: str) -> str:
        base = self.base_url.rstrip("/")
        if self.provider_id == "cloudflare":
            account_id = "default"
            if ":" in api_key:
                account_id, _ = api_key.split(":", 1)
            elif os.getenv("CLOUDFLARE_ACCOUNT_ID"):
                account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "default")
            base = base.replace("{account_id}", account_id.strip())
        return f"{base}/chat/completions"

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LLMRouter/1.0 (Windows NT 10.0; Win64; x64)",
        }
        if api_key and api_key != "no-key":
            actual_key = api_key.strip()
            if self.provider_id == "cloudflare" and ":" in actual_key:
                _, actual_key = actual_key.split(":", 1)
                actual_key = actual_key.strip()
            headers["Authorization"] = f"Bearer {actual_key}"

        if self.provider_id == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/shaurya-gupta-dev/LLM-Router"
            headers["X-Title"] = "LLM-Router"
        return headers

    def _handle_error_response(self, resp: httpx.Response):
        status = resp.status_code
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message") or str(body)
        except Exception:
            msg = resp.text or f"HTTP {status}"

        if status == 429:
            retry_after = 60.0
            ra_hdr = resp.headers.get("retry-after")
            if ra_hdr:
                try:
                    retry_after = float(ra_hdr)
                except ValueError:
                    pass
            raise RateLimitException(
                f"Rate limit exceeded on provider '{self.provider_id}': {msg}",
                provider=self.provider_id,
                retry_after=retry_after,
            )
        elif status in (401, 403):
            raise AuthException(
                f"Authentication failed on provider '{self.provider_id}': {msg}",
                provider=self.provider_id,
            )
        else:
            raise ProviderException(
                f"Provider '{self.provider_id}' returned {status}: {msg}",
                status_code=status,
                provider=self.provider_id,
            )

    def _prepare_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if self.provider_id == "aihorde":
            payload["max_tokens"] = max(16, max_tokens) if max_tokens else 512
        else:
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if tools:
                payload["tools"] = tools
                if self.provider_id in ("nvidia", "radeon"):
                    payload["parallel_tool_calls"] = False
        return payload

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        api_key: str,
    ) -> UnifiedChatResult:
        url = self._resolve_url(api_key)
        headers = self._build_headers(api_key)
        payload = self._prepare_payload(model, messages, tools, max_tokens, temperature, stream=False)

        try:
            resp = await self.client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            raise ProviderException(
                f"Network connection failed to '{self.provider_id}': {exc}",
                provider=self.provider_id,
            )

        if resp.status_code != 200:
            self._handle_error_response(resp)

        try:
            data = resp.json()
        except Exception:
            raise ProviderException(
                f"Provider '{self.provider_id}' returned invalid non-JSON body: {resp.text[:200]}",
                provider=self.provider_id,
            )
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        # Strip <think>...</think> blocks (DeepSeek-R1, QwQ via OpenRouter)
        if content:
            content = _THINK_TAG_RE.sub('', content).strip() or None
        else:
            content = None
        # NOTE: Do NOT promote 'reasoning' or 'reasoning_content' as content —
        # those fields contain internal chain-of-thought and must never be shown.
        tool_calls = msg.get("tool_calls")
        stop_reason = choice.get("finish_reason") or "stop"

        usage = data.get("usage", {})
        p_tokens = usage.get("prompt_tokens", 0) or 0
        c_tokens = usage.get("completion_tokens", 0) or 0

        return UnifiedChatResult(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            raw_response=data,
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
        url = self._resolve_url(api_key)
        headers = self._build_headers(api_key)
        payload = self._prepare_payload(model, messages, tools, max_tokens, temperature, stream=True)

        req = self.client.build_request("POST", url, headers=headers, json=payload)
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.RequestError as exc:
            raise ProviderException(
                f"Network connection failed to '{self.provider_id}': {exc}",
                provider=self.provider_id,
            )

        if resp.status_code != 200:
            # Must read error content before handling
            await resp.aread()
            self._handle_error_response(resp)

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except Exception:
                continue

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Check for text delta; skip reasoning-only deltas (internal thinking)
            text = delta.get("content")
            if text:
                # Strip any <think> prefix that arrives at stream start
                text = _THINK_TAG_RE.sub('', text)
            if text:
                yield {"text": text}
            # NOTE: Never yield delta.get("reasoning") — that is internal chain-of-thought.

            # Check for tool call deltas
            tc_deltas = delta.get("tool_calls")
            if tc_deltas:
                for tc in tc_deltas:
                    fn = tc.get("function", {})
                    yield {
                        "tool_call": {
                            "index": tc.get("index", 0),
                            "id": tc.get("id"),
                            "name": fn.get("name"),
                            "arguments_delta": fn.get("arguments", ""),
                        }
                    }

            if finish_reason:
                yield {"stop_reason": finish_reason}
