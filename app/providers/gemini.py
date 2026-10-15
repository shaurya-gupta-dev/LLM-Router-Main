"""Google AI Studio (Gemini) REST API provider adapter."""

from __future__ import annotations

import json
import logging
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

# Heuristic patterns: Gemma/Gemini models sometimes embed internal reasoning
# directly in the response text before the actual answer. We strip these.

# Pattern 1: explicit XML-style thinking wrappers (<think>, [thinking], etc.)
_THINK_TAG_PATTERNS = [
    re.compile(r'<think>.*?</think>\s*', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[thinking\].*?\[/thinking\]\s*', re.DOTALL | re.IGNORECASE),
]

# Pattern 2: multi-sentence inline reasoning preamble — starts with a
# reasoning-opener (e.g. "The user just said...") and ends right before the
# actual answer sentence begins.  We anchor to common openers and match
# greedily until the last ". " or ".\n" transition, then the answer begins.
_PREAMBLE_OPENER_RE = re.compile(
    r'^(?:The user|Okay|Alright|Sure,|Let me|I need to|According to|I should|I will)',
    re.IGNORECASE,
)


def _strip_thinking_preamble(text: str) -> str:
    """Remove internal reasoning/thinking preamble from model response text."""
    if not text:
        return text
    original = text

    # Step 1: strip explicit tag wrappers
    for pattern in _THINK_TAG_PATTERNS:
        text = pattern.sub('', text, count=1)

    # Step 2: strip multi-sentence inline preamble heuristically.
    # Only attempt if the text starts with a known reasoning-opener phrase.
    if _PREAMBLE_OPENER_RE.match(text):
        # Split on sentence boundaries: ". " or ".\n" keeping the delimiter.
        # Walk backwards from the end — the *last* sentence that ends with a
        # period followed immediately by an uppercase letter (no space) is
        # the preamble/answer boundary produced by models that run them together.
        # We find the index of the *last* ". " or ".\n" that is followed by
        # an uppercase letter and use everything after it as the real answer.
        m = None
        # Try to find transition: ". [A-Z]" or ".\n[A-Z]" (Gemma joins last preamble
        # sentence directly to the answer without a space / newline sometimes)
        for m in re.finditer(r'\.(?=\s*[A-Z])', text):
            pass  # Keep iterating to find the last one
        if m:
            candidate = text[m.end():].lstrip()
            # Safety: only strip if the candidate is shorter than the original
            # and looks like a real response (not just a fragment)
            if candidate and len(candidate) < len(text) * 0.8:
                text = candidate

    if text != original:
        logger.debug(
            "Stripped thinking preamble from response (len %d -> %d)", len(original), len(text)
        )
    return text.strip()


class GeminiProvider(BaseProvider):
    def __init__(self, provider_id: str = "gemini", base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        super().__init__(provider_id, base_url)
        use_http2 = False
        try:
            import h2  # noqa: F401
            use_http2 = True
        except ImportError:
            pass
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=90.0, write=15.0, pool=30.0),
            http2=use_http2,
            follow_redirects=True,
        )

    def _convert_messages_to_gemini(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Separate system instruction and translate message history into Gemini contents format."""
        system_instruction = None
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if role == "system":
                sys_text = content if isinstance(content, str) else json.dumps(content)
                system_instruction = {"parts": [{"text": sys_text}]}
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts: List[Dict[str, Any]] = []

            # Handle tool call responses (role: tool)
            if role == "tool":
                gemini_role = "user"
                call_id = msg.get("tool_call_id", "function_call")
                parts.append({
                    "functionResponse": {
                        "name": call_id,
                        "response": {"content": content or ""},
                    }
                })
            elif msg.get("tool_calls"):
                # Assistant message containing function calls
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args_dict = {}
                    parts.append({
                        "functionCall": {
                            "name": fn.get("name", ""),
                            "args": args_dict,
                        }
                    })
                if content:
                    parts.insert(0, {"text": content})
            elif isinstance(content, str) and content:
                parts.append({"text": content})
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            parts.append({"text": b.get("text", "")})
                        elif b.get("type") == "image_url":
                            # Vision support could be expanded here
                            pass
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        return system_instruction, contents

    @staticmethod
    def _clean_gemini_schema(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return schema
        disallowed_keys = {
            "$schema", "$id", "$ref", "$comment", "title", "additionalProperties",
            "propertyNames", "exclusiveMinimum", "exclusiveMaximum", "const",
            "patternProperties", "dependencies", "dependentRequired", "dependentSchemas",
            "unevaluatedProperties", "unevaluatedItems", "minProperties", "maxProperties",
        }
        cleaned = {}
        for k, v in schema.items():
            if not k or not str(k).strip() or str(k).startswith("$") or k in disallowed_keys:
                continue
            if k == "properties" and isinstance(v, dict):
                cleaned[k] = {pk: GeminiProvider._clean_gemini_schema(pv) for pk, pv in v.items() if pk and str(pk).strip()}
            elif k == "items" and isinstance(v, dict):
                cleaned[k] = GeminiProvider._clean_gemini_schema(v)
            elif k == "items" and isinstance(v, list):
                cleaned[k] = [GeminiProvider._clean_gemini_schema(item) for item in v]
            elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                cleaned[k] = [GeminiProvider._clean_gemini_schema(item) for item in v]
            elif isinstance(v, dict):
                cleaned[k] = GeminiProvider._clean_gemini_schema(v)
            elif isinstance(v, list):
                cleaned[k] = [GeminiProvider._clean_gemini_schema(item) if isinstance(item, (dict, list)) else item for item in v]
            else:
                cleaned[k] = v
        return cleaned

    def _convert_tools_to_gemini(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        declarations = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if "function" in t:
                fn = t["function"]
                params = self._clean_gemini_schema(fn.get("parameters", {}))
                declarations.append({
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": params,
                })
            elif "name" in t:
                raw_params = t.get("input_schema") or t.get("parameters", {})
                params = self._clean_gemini_schema(raw_params)
                declarations.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": params,
                })
        return [{"functionDeclarations": declarations}] if declarations else None

    def _handle_gemini_error(self, status: int, body_str: str):
        try:
            err_data = json.loads(body_str)
            err_obj = err_data.get("error", {})
            msg = err_obj.get("message") or body_str
            status_code_str = err_obj.get("status")
        except Exception:
            msg = body_str
            status_code_str = ""

        if status == 429 or "RESOURCE_EXHAUSTED" in status_code_str or "quota" in msg.lower():
            raise RateLimitException(
                f"Google Gemini rate limit: {msg}",
                provider="gemini",
                retry_after=60.0,
            )
        elif status in (401, 403) or "API_KEY_INVALID" in status_code_str:
            raise AuthException(
                f"Google Gemini authentication failed: {msg}",
                provider="gemini",
            )
        else:
            raise ProviderException(
                f"Google Gemini error ({status}): {msg}",
                status_code=status,
                provider="gemini",
            )

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        api_key: str,
    ) -> UnifiedChatResult:
        # Strip provider prefix if present
        clean_model = model.replace("models/", "")
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
            resp = await self.client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise ProviderException(f"Network error connecting to Gemini: {exc}", provider="gemini")

        if resp.status_code != 200:
            self._handle_gemini_error(resp.status_code, resp.text)

        data = resp.json()
        candidates = data.get("candidates") or [{}]
        cand = candidates[0]
        content_obj = cand.get("content", {})
        parts = content_obj.get("parts") or []

        text_parts = []
        tool_calls = []
        for p in parts:
            # Skip parts explicitly flagged as 'thought' (native Gemini thinking tokens)
            if p.get("thought") is True:
                continue
            if "text" in p:
                text_parts.append(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append({
                    "id": f"call_{fc.get('name')}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                })

        stop_reason = "stop"
        if cand.get("finishReason") == "MAX_TOKENS":
            stop_reason = "length"
        elif tool_calls:
            stop_reason = "tool_calls"

        usage = data.get("usageMetadata", {})
        p_tokens = usage.get("promptTokenCount", 0)
        c_tokens = usage.get("candidatesTokenCount", 0)

        raw_content = "\n".join(text_parts) if text_parts else None
        clean_content = _strip_thinking_preamble(raw_content) if raw_content else None

        return UnifiedChatResult(
            content=clean_content,
            tool_calls=tool_calls if tool_calls else None,
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
        clean_model = model.replace("models/", "")
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

        req = self.client.build_request("POST", url, json=payload)
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.RequestError as exc:
            raise ProviderException(f"Network error connecting to Gemini: {exc}", provider="gemini")

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

            cand = candidates[0]
            parts = cand.get("content", {}).get("parts") or []
            first_text_chunk = True
            for p in parts:
                # Skip parts explicitly flagged as 'thought' (native thinking tokens)
                if p.get("thought") is True:
                    continue
                if "text" in p:
                    chunk_text = p["text"]
                    # Only attempt heuristic stripping on the very first text chunk
                    if first_text_chunk:
                        chunk_text = _strip_thinking_preamble(chunk_text)
                        first_text_chunk = False
                    if chunk_text:
                        yield {"text": chunk_text}
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    yield {
                        "tool_call": {
                            "index": 0,
                            "id": f"call_{fc.get('name')}",
                            "name": fc.get("name"),
                            "arguments_delta": json.dumps(fc.get("args", {})),
                        }
                    }

            finish_reason = cand.get("finishReason")
            if finish_reason:
                sr = "stop"
                if finish_reason == "MAX_TOKENS":
                    sr = "length"
                yield {"stop_reason": sr}
