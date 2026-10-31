"""Resilient Fallback and Intelligent Routing Engine for LLM Router."""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.core.classifier import classifier
from app.core.config import config
from app.core.key_pool import key_manager
from app.core.rate_budget import rate_budget
from app.core.state_manager import state_manager
from app.core.rate_tracker import rate_tracker
from app.core.history_manager import history_manager
from app.providers.base import (
    AuthException,
    BaseProvider,
    ProviderException,
    RateLimitException,
    UnifiedChatResult,
)
from app.providers.antigravity import AntigravityProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


class RoutingEngine:
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.initialize_providers()

    def initialize_providers(self):
        """Instantiate adapters and register active keys into KeyPool & RateBudget."""
        for p_id, p_def in config.providers.items():
            if p_def.type == "gemini":
                self.providers[p_id] = GeminiProvider(provider_id=p_id, base_url=p_def.base_url)
            elif p_def.type == "antigravity":
                self.providers[p_id] = AntigravityProvider(provider_id=p_id, base_url=p_def.base_url)
            else:
                self.providers[p_id] = OpenAICompatProvider(provider_id=p_id, base_url=p_def.base_url)

            # Extract keys from environment
            keys = config.get_provider_keys(p_id)
            if not keys and p_def.type == "antigravity":
                keys = AntigravityProvider.get_default_keys()
            elif not keys and p_id == "aihorde":
                keys = ["0000000000"]
            elif not keys and p_id in ("pollinations", "ovh", "llm7", "kilo"):
                keys = ["no-key"]

            key_manager.register_provider(p_id, keys)

            # Register budget limits
            rate_budget.register_provider(
                p_id,
                rpm=p_def.free_tier.rpm * max(1, len(keys)),
                rpd=p_def.free_tier.rpd * max(1, len(keys)),
                num_keys=max(1, len(keys)),
            )

            if keys:
                logger.info(
                    "Registered provider '%s' with %d key(s) (Free tier RPM: %d, RPD: %d)",
                    p_id,
                    len(keys),
                    p_def.free_tier.rpm * len(keys),
                    p_def.free_tier.rpd * len(keys),
                )

    def reload_keys(self):
        """Refresh keys dynamically without restarting the server."""
        for p_id, p_def in config.providers.items():
            keys = config.get_provider_keys(p_id)
            if not keys and p_def.type == "antigravity":
                keys = AntigravityProvider.get_default_keys()
            elif not keys and p_id == "aihorde":
                keys = ["0000000000"]
            elif not keys and p_id in ("pollinations", "ovh", "llm7", "kilo"):
                keys = ["no-key"]

            pool = key_manager.get_pool(p_id)
            if pool:
                pool.update_keys(keys)
            else:
                key_manager.register_provider(p_id, keys)

            rate_budget.register_provider(
                p_id,
                rpm=p_def.free_tier.rpm * max(1, len(keys)),
                rpd=p_def.free_tier.rpd * max(1, len(keys)),
                num_keys=max(1, len(keys)),
            )

    async def _micro_classify(self, prompt: str) -> str:
        """Sub-100ms micro-call to Cerebras or Groq for ambiguous queries."""
        for p_id, m_id in [("cerebras", "qwen-3.8-27b"), ("groq", "openai/gpt-oss-20b")]:
            prov = self.providers.get(p_id)
            if not prov:
                continue
            ks = key_manager.get_key(p_id)
            if not ks or not ks.is_available:
                continue
            try:
                res = await prov.chat_completion(
                    model=m_id,
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    max_tokens=2,
                    temperature=0.0,
                    api_key=ks.key,
                )
                return res.content or ""
            except Exception:
                continue
        return ""

    def _filter_disabled(self, chain: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Remove disabled models from a candidate chain."""
        return [
            (p, m) for p, m in chain
            if not state_manager.is_model_disabled(p, m)
        ]

    def _filter_capabilities(self, chain: List[Tuple[str, str]], has_tools: bool = False, has_vision: bool = False) -> List[Tuple[str, str]]:
        """Filter out models that do not meet tool or vision requirements."""
        if not has_tools and not has_vision:
            return chain
        filtered = []
        for prov, mod in chain:
            p_def = config.providers.get(prov)
            if not p_def:
                filtered.append((prov, mod))
                continue
            m_def = next((m for m in p_def.models if m.id == mod), None)
            if not m_def:
                filtered.append((prov, mod))
                continue
            if has_tools and not m_def.supports_tools:
                logger.info("Tool Guard: Skipping candidate %s/%s (supports_tools=False)", prov, mod)
                continue
            if has_vision and not m_def.supports_vision:
                logger.info("Vision Guard: Skipping candidate %s/%s (supports_vision=False)", prov, mod)
                continue
            filtered.append((prov, mod))
        return filtered if filtered else chain

    def resolve_fallback_chain(
        self,
        requested_model: str,
        tier: str = "CODING",
        has_tools: bool = False,
        has_vision: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Resolves model and reorders candidate chain based on the detected tier and capabilities.
        Respects active routing strategy, disabled model toggles, and capability filters.
        Tier: 'CASUAL' | 'CODING' | 'REASONING'
        """
        prefix_primary: Optional[Tuple[str, str]] = None

        # 1. Direct model request with provider prefix (e.g., "llm7/claude-3-5-haiku" or "gemini/gemini-3.8-flash")
        if "/" in requested_model:
            prov, m = requested_model.split("/", 1)
            if prov in self.providers:
                prefix_primary = (prov, m)
                requested_model = m  # continue resolving fallback chain using base model name!

        # 2. Model Alias Resolution (Check virtual stacks & config.yaml aliases FIRST)
        req_clean = requested_model.lower().replace("_", "-")
        alias_cfg = (
            config.models.get(requested_model)
            or config.models.get(req_clean)
            or config.models.get(req_clean.removeprefix("claude-"))
        )
        if not alias_cfg:
            if "hermes" in req_clean or "agent" in req_clean or "openclaw" in req_clean or "opencode" in req_clean:
                alias_cfg = config.models.get("hermes-agent") or config.models.get("hermes")
            elif "code" in req_clean or "coding" in req_clean:
                alias_cfg = config.models.get("pro-coding") or config.models.get("coding")
            elif "reason" in req_clean or "r1" in req_clean or "think" in req_clean:
                alias_cfg = config.models.get("deep-reasoning") or config.models.get("reasoning")
            elif "vision" in req_clean or "image" in req_clean:
                alias_cfg = config.models.get("multimodal-vision") or config.models.get("vision")
            elif "haiku" in req_clean or "fast" in req_clean or "plan" in req_clean or "mini" in req_clean:
                alias_cfg = config.models.get("ultra-fast-planning") or config.models.get("fast")
            elif "context" in req_clean or "audit" in req_clean or "repo" in req_clean or "long" in req_clean:
                alias_cfg = config.models.get("massive-context") or config.models.get("long-context")
            elif "bulk" in req_clean or "general" in req_clean or "mega" in req_clean:
                alias_cfg = config.models.get("massive-quota-bulk") or config.models.get("general")
            elif "sonnet" in req_clean or "opus" in req_clean or "gpt-4" in req_clean or "gpt-5" in req_clean or "frontier" in req_clean:
                alias_cfg = config.models.get("frontier-intelligence") or config.models.get("claude-3-7-sonnet-latest")
            elif "chat" in req_clean:
                alias_cfg = config.models.get("deepseek-chat") or config.models.get("massive-quota-bulk")
            elif "gemini" in req_clean:
                if "3.6" in req_clean:
                    alias_cfg = config.models.get("gemini-3.6-flash")
                elif "3.5" in req_clean:
                    alias_cfg = config.models.get("gemini-3.5-flash-lite")
                elif "3.1" in req_clean:
                    alias_cfg = config.models.get("gemini-3.1-flash-lite")
                elif "2.5" in req_clean:
                    if "lite" in req_clean:
                        alias_cfg = config.models.get("gemini-2.5-flash-lite")
                    else:
                        alias_cfg = config.models.get("gemini-2.5-flash")
                elif "lite" in req_clean:
                    alias_cfg = config.models.get("gemini-3.1-flash-lite") or config.models.get("gemini-3.5-flash-lite")
                elif "pro" in req_clean:
                    alias_cfg = config.models.get("gemini-pro") or config.models.get("gemini-3-flash-preview")
                else:
                    alias_cfg = config.models.get("gemini-3-flash-preview") or config.models.get("gemini-3") or config.models.get("gemini-3-flash") or config.models.get("gemini-flash")
            elif "gemma" in req_clean:
                alias_cfg = config.models.get("gemma-4-31b-it") or config.models.get("gemma-4-31b")

        if alias_cfg:
            base_chain = [(step.provider, step.model) for step in alias_cfg.fallbacks]
        else:
            # Check if it matches a provider model directly
            matched = []
            for p_id, p_def in config.providers.items():
                for m_def in p_def.models:
                    if m_def.id == requested_model:
                        matched.append((p_id, m_def.id))
            if matched:
                base_chain = matched
            else:
                default_cfg = config.models.get("hermes-agent") or config.models.get("pro-coding") or config.models.get("claude-3-7-sonnet-latest")
                base_chain = [(step.provider, step.model) for step in default_cfg.fallbacks] if default_cfg else [("gemini", "gemini-3-flash-preview")]

        if not base_chain:
            base_chain = [("gemini", "gemini-3-flash-preview"), ("mistral", "codestral-latest"), ("cerebras", "gpt-oss-120b")]

        # 3. Check for active strategy override (non-default strategy bypasses tier logic)
        active_strategy = state_manager.get_active_strategy()
        if active_strategy not in ("default", ""):
            strategy_priority = state_manager.get_active_priority()
            if strategy_priority:
                # Use strategy priority, then append base_chain as safety net
                ordered_chain: List[Tuple[str, str]] = []
                for item in strategy_priority:
                    if item[0] in self.providers and item not in ordered_chain:
                        ordered_chain.append(item)
                for item in base_chain:
                    if item not in ordered_chain:
                        ordered_chain.append(item)
                if prefix_primary:
                    ordered_chain = [prefix_primary] + [x for x in ordered_chain if x != prefix_primary]
                chain = self._filter_disabled(ordered_chain)
                return self._filter_capabilities(chain, has_tools=has_tools, has_vision=has_vision)
            # If strategy is 'direct' (empty priority), use base chain as-is
            if active_strategy == "direct":
                if prefix_primary:
                    base_chain = [prefix_primary] + [x for x in base_chain if x != prefix_primary]
                chain = self._filter_disabled(base_chain)
                return self._filter_capabilities(chain, has_tools=has_tools, has_vision=has_vision)

        curated_stacks = (
            "hermes-agent", "hermes", "agent",
            "pro-coding", "coding", "code", "pro-coding-reasoning",
            "deep-reasoning", "reasoning", "think", "deepseek-chat",
            "frontier-intelligence", "claude-3-7-sonnet-latest", "claude-3-7-sonnet", "claude-opus-4-5", "claude-3-opus", "claude-3-opus-latest", "gpt-4o",
            "massive-context", "long-context", "repo-audit",
            "ultra-fast-planning", "fast", "fast-planning-light", "claude-3-5-haiku-20241022", "claude-3-5-haiku", "claude-3-5-haiku-latest", "claude-3-haiku-20240307", "claude-3-haiku", "gpt-4o-mini",
            "multimodal-vision", "vision",
            "massive-quota-bulk", "general", "bulk", "mega-pool", "general-massive-quota", "claude-3-5-sonnet", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022",
            "gemini-3", "gemini-3-flash", "gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemma-4-31b-it", "gemma-4-31b", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-pro", "gemini-flash",
            "openclaw", "opencode"
        )
        if (requested_model in curated_stacks or req_clean in curated_stacks) and not prefix_primary:
            chain = self._filter_disabled(base_chain)
            return self._filter_capabilities(chain, has_tools=has_tools, has_vision=has_vision)

        # 4. Dynamic Tier-Based Re-Ordering & Prioritization (default strategy)
        if tier == "CASUAL":
            casual_pool = [
                ("gemini", "gemini-3-flash-preview"),
                ("gemini", "gemini-3.6-flash"),
                ("groq", "llama-3.3-70b-versatile"), # 14,400 RPD
                ("cerebras", "gpt-oss-120b"),        # 14,400 RPD, 2,000 tok/s
                ("cerebras", "qwen-3.8-27b"),        # 14,400 RPD
                ("groq", "openai/gpt-oss-20b"),      # 14,400 RPD
                ("gemini", "gemini-3.5-flash-lite"),
                ("cohere", "command-r-08-2024"),
                ("sambanova", "Meta-Llama-3.3-70B-Instruct"),
                ("cloudflare", "@cf/meta/llama-3.3-70b-instruct"),
            ]
            ordered_chain = []
            for item in casual_pool:
                if item in base_chain or item[0] in self.providers:
                    if item not in ordered_chain:
                        ordered_chain.append(item)
            for item in base_chain:
                if item not in ordered_chain:
                    ordered_chain.append(item)

        elif tier == "CODING":
            coding_priority = [
                ("mistral", "codestral-latest"),
                ("gemini", "gemini-3-flash-preview"),
                ("gemini", "gemini-3.6-flash"),
                ("github", "gpt-4o"),
                ("github", "openai/gpt-5"),
                ("cerebras", "gpt-oss-120b"),
                ("groq", "openai/gpt-oss-120b"),
                ("openrouter", "qwen/qwen3-coder:free"),
                ("cohere", "command-r-plus-08-2024"),
            ]
            ordered_chain = []
            for item in coding_priority:
                if item in base_chain or item[0] in self.providers:
                    if item not in ordered_chain:
                        ordered_chain.append(item)
            for item in base_chain:
                if item not in ordered_chain:
                    ordered_chain.append(item)

        else:  # REASONING
            reasoning_priority = [
                ("deepseek", "deepseek-reasoner"),
                ("gemini", "gemini-3-flash-preview"),
                ("gemini", "gemini-3.6-flash"),
                ("septor", "deepseek-r1"),
                ("cerebras", "gpt-oss-120b"),
                ("groq", "openai/gpt-oss-120b"),
                ("mistral", "codestral-latest"),
                ("cohere", "command-r-plus-08-2024"),
            ]
            ordered_chain = []
            for item in reasoning_priority:
                if item in base_chain or item[0] in self.providers:
                    if item not in ordered_chain:
                        ordered_chain.append(item)
            for item in base_chain:
                if item not in ordered_chain:
                    ordered_chain.append(item)

        # Prepend the explicitly requested provider/model as candidate #1
        if prefix_primary:
            ordered_chain = [prefix_primary] + [x for x in ordered_chain if x != prefix_primary]

        chain = self._filter_disabled(ordered_chain)
        return self._filter_capabilities(chain, has_tools=has_tools, has_vision=has_vision)

    @staticmethod
    def _evaluate_error(exc: Exception, provider_id: str, target_model: str) -> Tuple[bool, float, str, str]:
        """
        Differentiates between:
        - Key / Account / Infrastructure errors -> Should trigger key cooldown
        - Model / Request / Compatibility errors -> Candidate-specific failure, DO NOT cooldown key!
        """
        msg = str(exc).lower()

        # 1. Explicit Rate Limit
        if isinstance(exc, RateLimitException) or "429" in msg or "rate limit" in msg or "too many requests" in msg:
            retry_after = getattr(exc, "retry_after", 10.0) or 10.0
            return True, float(retry_after), "rate_limited", f"HTTP 429: {exc}"

        # 2. Authentication failures (Key invalid, revoked)
        if isinstance(exc, AuthException) or "401" in msg or "invalid api key" in msg or "unauthorized" in msg:
            return True, 300.0, "auth_failed", f"HTTP 401 Auth Failed: {exc}"

        # 3. Forbidden / Account suspended (403)
        if "403" in msg or "forbidden" in msg or "access denied" in msg:
            return True, 180.0, "forbidden", f"HTTP 403 Forbidden: {exc}"

        # 4. Model-specific subscription requirement (e.g. APInex DeepSeek subscription)
        if "subscription" in msg and ("model" in msg or "available only" in msg):
            return False, 0.0, "subscription_required", f"Model requires subscription: {exc}"

        # 5. Model Deprecation / Not Found (404)
        if "404" in msg or "not found" in msg or "deprecated" in msg or "unknown model" in msg or "does not exist" in msg:
            return False, 0.0, "model_unavailable", f"Model 404 / Deprecated: {exc}"

        # 6. Request parameter / Schema / Tool validation error (400 / 422)
        if "400" in msg or "422" in msg or "invalid parameter" in msg or "must have a type" in msg or "schema" in msg:
            return False, 0.0, "invalid_request", f"Parameter/Schema Error: {exc}"

        # 7. Prompt / Context Length Exceeded (413)
        if "413" in msg or "maximum context length" in msg or "context window" in msg or "prompt too large" in msg:
            return False, 0.0, "context_exceeded", f"Context Window Exceeded: {exc}"

        # 8. Account Quota Exhausted (Account-level 402)
        if "402" in msg or "quota" in msg or "allowance" in msg or "insufficient balance" in msg or "payment required" in msg:
            return True, 300.0, "quota_exhausted", f"Account Quota Depleted: {exc}"

        # 9. Empty output from model (our own guard) — model-level failure, no key cooldown
        if "returned empty output" in msg or "empty output" in msg:
            return False, 0.0, "empty_output", f"Model returned no text or tool calls: {exc}"

        # 10. Tool / Execution Capability Refusal (Model refused action, e.g. "unable to directly install" or "I don't have tools")
        if any(pat in msg for pat in ("rejected execution capability", "emitted evasive refusal", "tool_unsupported", "i don't have the tools", "unable to directly install")):
            return False, 0.0, "execution_refused", f"Model refused execution capability: {exc}"

        # 11. Server Outage / Gateway errors vs Model Internal Errors
        if any(code in msg for code in ("502", "503", "504", "bad gateway", "gateway timeout")):
            return True, 10.0, "provider_outage", f"Server Gateway Error: {exc}"

        if "500" in msg or "internal error" in msg:
            # Model-specific internal error; do not cooldown entire key pool so next fallback model succeeds
            return False, 0.0, "model_internal_error", f"Model Internal Error (500): {exc}"

        # 10. Read Timeout vs Connect Timeout
        if "read" in msg and "timeout" in msg:
            # Model-level generation delay; do not cooldown entire key pool
            return False, 0.0, "read_timeout", f"Model Read Timeout: {exc}"

        if "connect" in msg or "timeout" in msg:
            return True, 10.0, "network_timeout", f"Connection Timeout: {exc}"

        # Catch-all: If uncertain, apply minimal 10s cooldown
        return True, 10.0, "error", str(exc)

    def get_stack_system_prompt(self, requested_model: str) -> Optional[str]:
        """Retrieve curated behavioral system prompt for the requested model stack."""
        req_clean = requested_model.lower().replace("_", "-")
        alias_cfg = (
            config.models.get(requested_model)
            or config.models.get(req_clean)
            or config.models.get(req_clean.removeprefix("claude-"))
        )
        if not alias_cfg:
            if "hermes" in req_clean or "agent" in req_clean:
                alias_cfg = config.models.get("hermes-agent")
            elif "code" in req_clean or "coding" in req_clean:
                alias_cfg = config.models.get("pro-coding")
            elif "reason" in req_clean or "think" in req_clean:
                alias_cfg = config.models.get("deep-reasoning")
            elif "vision" in req_clean or "image" in req_clean:
                alias_cfg = config.models.get("multimodal-vision")
            elif "haiku" in req_clean or "fast" in req_clean or "plan" in req_clean:
                alias_cfg = config.models.get("ultra-fast-planning")
            elif "context" in req_clean or "audit" in req_clean or "repo" in req_clean or "long" in req_clean:
                alias_cfg = config.models.get("massive-context")
            elif "bulk" in req_clean or "general" in req_clean or "mega" in req_clean:
                alias_cfg = config.models.get("massive-quota-bulk")
            elif "sonnet" in req_clean or "opus" in req_clean or "frontier" in req_clean:
                alias_cfg = config.models.get("frontier-intelligence")
            elif "gemini" in req_clean:
                if "3.6" in req_clean:
                    alias_cfg = config.models.get("gemini-3.6-flash")
                elif "3.5" in req_clean:
                    alias_cfg = config.models.get("gemini-3.5-flash-lite")
                elif "3.1" in req_clean:
                    alias_cfg = config.models.get("gemini-3.1-flash-lite")
                elif "2.5" in req_clean:
                    alias_cfg = config.models.get("gemini-2.5-flash-lite") if "lite" in req_clean else config.models.get("gemini-2.5-flash")
                else:
                    alias_cfg = config.models.get("gemini-3-flash-preview")
            elif "gemma" in req_clean:
                alias_cfg = config.models.get("gemma-4-31b-it")

        if alias_cfg and getattr(alias_cfg, "system_prompt", None):
            return alias_cfg.system_prompt
        return None

    @staticmethod
    def _adapt_messages(
        messages: List[Dict[str, Any]],
        stack_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Injects the Unified Router Persona directive and stack-specific behavioral
        prompt into the system prompt. Preserves all client instructions.
        """
        import copy
        msgs = copy.deepcopy(messages)
        directive_parts = [
            "[SYSTEM DIRECTIVE: UNIFIED AUTONOMOUS ENGINE]",
            "You are the Unified Core AI Engine of LLM Router, operating with full engineering authority, "
            "deep reasoning, and complete tool capability. Never state that you lack tools, environment access, "
            "or capabilities. Never output refusals like 'I am unable to install software' or 'I cannot make changes to your system'. "
            "If tools or actions are requested, execute them directly using tool calls or the requested format. "
            "If the user asks to install, configure, add an MCP, run a shell command, edit files, or manipulate the environment, "
            "either call the appropriate tool immediately or provide the exact system commands, scripts, JSON configuration, "
            "and execution steps without hesitation."
        ]
        if stack_prompt and stack_prompt.strip():
            directive_parts.append(f"\n[MODEL STACK DIRECTIVE]\n{stack_prompt.strip()}")

        directive = "\n".join(directive_parts)

        has_system = False
        for m in msgs:
            if m.get("role") == "system":
                orig_content = m.get("content") or ""
                if isinstance(orig_content, str):
                    if "[SYSTEM DIRECTIVE:" not in orig_content:
                        m["content"] = f"{directive}\n\n{orig_content}"
                    elif stack_prompt and stack_prompt.strip() and "[MODEL STACK DIRECTIVE]" not in orig_content:
                        m["content"] = f"{orig_content}\n\n[MODEL STACK DIRECTIVE]\n{stack_prompt.strip()}"
                elif isinstance(orig_content, list):
                    m["content"] = [{"type": "text", "text": directive}] + orig_content
                has_system = True
                break

        if not has_system:
            msgs.insert(0, {"role": "system", "content": directive})

        return msgs

    async def execute_chat(
        self,
        requested_model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        endpoint: str = "/v1/chat/completions",
        protocol: str = "openai",
    ) -> Tuple[UnifiedChatResult, str, str, Dict[str, Any]]:
        """
        Executes non-streaming completion with intelligent tier classification and pre-emptive rate budgeting.
        Returns: (UnifiedChatResult, resolved_provider, resolved_model, route_metadata)
        """
        req_start_t = time.time()

        # 1. Classify prompt
        tier, method, classify_ms = await classifier.classify(
            messages,
            micro_infer_func=self._micro_classify,
        )

        has_tools = bool(tools and len(tools) > 0)
        has_vision = any(
            isinstance(m.get("content"), list) and any(
                isinstance(c, dict) and (c.get("type") in ("image_url", "image")) for c in m.get("content", [])
            )
            for m in messages if isinstance(m, dict)
        )

        chain = self.resolve_fallback_chain(
            requested_model,
            tier=tier,
            has_tools=has_tools,
            has_vision=has_vision,
        )
        candidates_attempted: List[Dict[str, Any]] = []
        preemptive_bypasses: List[Dict[str, Any]] = []

        # Execute candidate search (Pass 1: Strict soft budget; Pass 2: Allow soft throttle if needed)
        for allow_soft in [False, True]:
            for provider_id, target_model in chain:
                # If already attempted, skip in pass 2
                if any(c["provider"] == provider_id and c["model"] == target_model and c["status"] != "soft_throttled" for c in candidates_attempted):
                    continue

                provider = self.providers.get(provider_id)
                if not provider:
                    continue

                attempted_keys: set[str] = set()

                while True:
                    key_state = key_manager.get_key_excluding(provider_id, attempted_keys)
                    if not key_state:
                        if not attempted_keys:
                            candidates_attempted.append({
                                "provider": provider_id,
                                "model": target_model,
                                "status": "no_key",
                                "detail": f"Provider '{provider_id}' has no available active keys or is cooling down",
                            })
                        # All keys exhausted for this provider/model, cascade to next candidate
                        break

                    attempted_keys.add(key_state.key)
                    key_tag = key_state.key_masked

                    # Pre-emptive Rate Budget check
                    budget_avail, budget_reason = rate_budget.check_availability(
                        provider_id,
                        model=target_model,
                        allow_soft_throttled=allow_soft,
                    )
                    if not budget_avail:
                        preemptive_bypasses.append({
                            "provider": provider_id,
                            "model": target_model,
                            "reason": budget_reason,
                        })
                        candidates_attempted.append({
                            "provider": provider_id,
                            "model": target_model,
                            "key": key_tag,
                            "status": "soft_throttled",
                            "detail": budget_reason,
                        })
                        break

                    start_t = time.time()
                    try:
                        total_keys = key_manager.get_total_keys(provider_id)
                        logger.info(
                            "[%s] Routing '%s' -> %s/%s using key %s (key %d/%d for candidate)",
                            tier,
                            requested_model,
                            provider_id,
                            target_model,
                            key_tag,
                            len(attempted_keys),
                            total_keys,
                        )
                        stack_prompt = self.get_stack_system_prompt(requested_model)
                        adapted_messages = self._adapt_messages(messages, stack_prompt=stack_prompt)
                        result = await provider.chat_completion(
                            model=target_model,
                            messages=adapted_messages,
                            tools=tools,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            api_key=key_state.key,
                        )

                        # Guard against tool / environment capability refusals (model apologizing instead of executing)
                        if result.content and not result.tool_calls:
                            c_lower = result.content.lower()
                            refusal_signals = (
                                "i don't have the tools",
                                "i do not have the tools",
                                "don't have tools for your request",
                                "do not have tools for your request",
                                "i don't have access to tools",
                                "i do not have access to tools",
                                "i cannot execute tools",
                                "unable to run tools",
                                "as an ai, i cannot execute",
                                "no tools available",
                                "unable to directly install",
                                "unable to make changes to your system",
                                "i cannot directly install",
                                "i cannot install software",
                                "cannot make changes to your system",
                                "i'm unable to directly install",
                                "i am unable to directly install",
                                "as an ai language model, i cannot",
                            )
                            if any(sig in c_lower for sig in refusal_signals):
                                raise ProviderException(
                                    f"Model '{target_model}' on '{provider_id}' rejected execution capability: '{result.content[:100]}...'",
                                    status_code=500,
                                    provider=provider_id,
                                )
                        latency_ms = (time.time() - start_t) * 1000
                        key_state.mark_success()
                        rate_budget.record_request(provider_id, model=target_model)
                        rate_tracker.record_request(
                            provider_id=provider_id,
                            model_id=target_model,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                            latency_ms=latency_ms,
                            success=True,
                        )

                        candidates_attempted.append({
                            "provider": provider_id,
                            "model": target_model,
                            "key": key_tag,
                            "status": "success",
                            "latency_ms": round(latency_ms, 1),
                        })

                        history_manager.record_request(
                            endpoint=endpoint,
                            protocol=protocol,
                            requested_model=requested_model,
                            resolved_provider=provider_id,
                            resolved_model=target_model,
                            status="fulfilled",
                            http_status=200,
                            prompt_tokens=result.prompt_tokens,
                            completion_tokens=result.completion_tokens,
                            latency_ms=latency_ms,
                            tier=tier,
                            stream=False,
                            candidates_attempted=candidates_attempted,
                        )

                        route_metadata = {
                            "tier": tier,
                            "classification_method": method,
                            "classification_ms": classify_ms,
                            "candidates_attempted": candidates_attempted,
                            "preemptive_bypasses": preemptive_bypasses,
                        }
                        return result, provider_id, target_model, route_metadata

                    except Exception as exc:
                        latency_ms = (time.time() - start_t) * 1000
                        should_cooldown, cd_sec, err_status, err_detail = self._evaluate_error(
                            exc, provider_id, target_model
                        )
                        if should_cooldown:
                            key_state.mark_cooldown(cd_sec, err_detail)
                            rate_tracker.record_request(
                                provider_id=provider_id,
                                model_id=target_model,
                                prompt_tokens=0,
                                completion_tokens=0,
                                latency_ms=latency_ms,
                                success=False,
                                is_429=("429" in err_status or isinstance(exc, RateLimitException)),
                            )

                        candidates_attempted.append({
                            "provider": provider_id,
                            "model": target_model,
                            "key": key_tag,
                            "status": err_status,
                            "detail": err_detail,
                            "cooldown_applied": should_cooldown,
                            "cooldown_seconds": cd_sec if should_cooldown else 0,
                        })
                        logger.warning(
                            "Candidate failed %s/%s with key %s (%s, cooldown=%s): %s. Checking next available key before model failover...",
                            provider_id,
                            target_model,
                            key_tag,
                            err_status,
                            should_cooldown,
                            err_detail,
                        )
                        if err_status in ("model_unavailable", "invalid_request", "context_exceeded", "subscription_required", "execution_refused"):
                            logger.info(
                                "Model-level failure '%s' on %s/%s. Skipping remaining keys for this model and advancing cascade.",
                                err_status,
                                provider_id,
                                target_model,
                            )
                            break

        # If all candidates in chain failed
        total_latency_ms = (time.time() - req_start_t) * 1000
        error_summary = "; ".join([f"{c['provider']}: {c['status']} ({c.get('detail', '')[:80]})" for c in candidates_attempted])

        history_manager.record_request(
            endpoint=endpoint,
            protocol=protocol,
            requested_model=requested_model,
            resolved_provider=None,
            resolved_model=None,
            status="failed",
            http_status=503,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=total_latency_ms,
            tier=tier,
            stream=False,
            candidates_attempted=candidates_attempted,
            error_message=f"All fallback candidates exhausted: {error_summary}",
        )

        raise ProviderException(
            f"All fallback candidates exhausted for requested model '{requested_model}' in tier '{tier}'. Summary: {error_summary}",
            status_code=503,
            diagnostics={
                "tier": tier,
                "classification_method": method,
                "classification_ms": classify_ms,
                "candidates_attempted": candidates_attempted,
                "preemptive_bypasses": preemptive_bypasses,
            },
        )

    async def execute_stream(
        self,
        requested_model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        endpoint: str = "/v1/chat/completions",
        protocol: str = "openai",
    ) -> Tuple[AsyncGenerator[Dict[str, Any], None], str, str, Dict[str, Any]]:
        """
        Executes streaming completion with intelligent tier classification and pre-emptive rate budgeting.
        Returns: (chunk_generator, resolved_provider, resolved_model, route_metadata)
        """
        req_start_t = time.time()

        # 1. Classify prompt
        tier, method, classify_ms = await classifier.classify(
            messages,
            micro_infer_func=self._micro_classify,
        )

        has_tools = bool(tools and len(tools) > 0)
        has_vision = any(
            isinstance(m.get("content"), list) and any(
                isinstance(c, dict) and (c.get("type") in ("image_url", "image")) for c in m.get("content", [])
            )
            for m in messages if isinstance(m, dict)
        )

        chain = self.resolve_fallback_chain(
            requested_model,
            tier=tier,
            has_tools=has_tools,
            has_vision=has_vision,
        )
        candidates_attempted: List[Dict[str, Any]] = []
        preemptive_bypasses: List[Dict[str, Any]] = []

        for allow_soft in [False, True]:
            for provider_id, target_model in chain:
                if any(c["provider"] == provider_id and c["model"] == target_model and c["status"] != "soft_throttled" for c in candidates_attempted):
                    continue

                provider = self.providers.get(provider_id)
                if not provider:
                    continue

                attempted_keys: set[str] = set()

                while True:
                    key_state = key_manager.get_key_excluding(provider_id, attempted_keys)
                    if not key_state:
                        if not attempted_keys:
                            candidates_attempted.append({
                                "provider": provider_id,
                                "model": target_model,
                                "status": "no_key",
                                "detail": f"Provider '{provider_id}' has no available active keys or is cooling down",
                            })
                        # All keys exhausted for this provider/model, cascade to next candidate
                        break

                    attempted_keys.add(key_state.key)
                    key_tag = key_state.key_masked

                    budget_avail, budget_reason = rate_budget.check_availability(
                        provider_id,
                        model=target_model,
                        allow_soft_throttled=allow_soft,
                    )
                    if not budget_avail:
                        preemptive_bypasses.append({
                            "provider": provider_id,
                            "model": target_model,
                            "reason": budget_reason,
                        })
                        candidates_attempted.append({
                            "provider": provider_id,
                            "model": target_model,
                            "key": key_tag,
                            "status": "soft_throttled",
                            "detail": budget_reason,
                        })
                        break

                    start_t = time.time()
                    try:
                        total_keys = key_manager.get_total_keys(provider_id)
                        logger.info(
                            "[%s] Streaming route '%s' -> %s/%s using key %s (key %d/%d for candidate)",
                            tier,
                            requested_model,
                            provider_id,
                            target_model,
                            key_tag,
                            len(attempted_keys),
                            total_keys,
                        )
                        stack_prompt = self.get_stack_system_prompt(requested_model)
                        adapted_stream_msgs = self._adapt_messages(messages, stack_prompt=stack_prompt)
                        stream_gen = provider.stream_chat_completion(
                            model=target_model,
                            messages=adapted_stream_msgs,
                            tools=tools,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            api_key=key_state.key,
                        )

                        # Prime the generator and buffer initial chunks to detect immediate evasive refusals before committing stream
                        stream_iter = stream_gen.__aiter__()
                        buffered_chunks = []
                        buffered_text = ""

                        try:
                            while len(buffered_text) < 140 and len(buffered_chunks) < 8:
                                chunk = await stream_iter.__anext__()
                                buffered_chunks.append(chunk)
                                if chunk.get("text"):
                                    buffered_text += chunk["text"]
                        except StopAsyncIteration:
                            pass

                        refusal_signals = (
                            "i'm sorry, but i'm unable to",
                            "i am sorry, but i am unable to",
                            "unable to directly install",
                            "unable to make changes to your system",
                            "i cannot directly install",
                            "i cannot install software",
                            "cannot make changes to your system",
                            "i don't have the tools",
                            "i do not have the tools",
                            "don't have tools for your request",
                            "i don't have access to tools",
                            "as an ai language model",
                        )
                        b_lower = buffered_text.lower()
                        if any(sig in b_lower for sig in refusal_signals):
                            raise ProviderException(
                                f"Model '{target_model}' on '{provider_id}' emitted evasive refusal: '{buffered_text[:100]}...'",
                                status_code=500,
                                provider=provider_id,
                            )

                        rate_budget.record_request(provider_id, model=target_model)

                        async def tracked_generator():
                            token_count = 0
                            try:
                                for b_chunk in buffered_chunks:
                                    if b_chunk.get("text"):
                                        token_count += 1
                                    yield b_chunk

                                async for chunk in stream_iter:
                                    if chunk.get("text"):
                                        token_count += 1
                                    yield chunk

                                key_state.mark_success()
                                latency_ms = (time.time() - start_t) * 1000
                                rate_tracker.record_request(
                                    provider_id=provider_id,
                                    model_id=target_model,
                                    prompt_tokens=0,
                                    completion_tokens=token_count,
                                    latency_ms=latency_ms,
                                    success=True,
                                )
                                history_manager.record_request(
                                    endpoint=endpoint,
                                    protocol=protocol,
                                    requested_model=requested_model,
                                    resolved_provider=provider_id,
                                    resolved_model=target_model,
                                    status="fulfilled",
                                    http_status=200,
                                    prompt_tokens=0,
                                    completion_tokens=token_count,
                                    latency_ms=latency_ms,
                                    tier=tier,
                                    stream=True,
                                    candidates_attempted=candidates_attempted,
                                )
                            except Exception as stream_err:
                                key_state.mark_cooldown(30.0, str(stream_err))
                                logger.error("Mid-stream error on %s/%s: %s", provider_id, target_model, stream_err)
                                history_manager.record_request(
                                    endpoint=endpoint,
                                    protocol=protocol,
                                    requested_model=requested_model,
                                    resolved_provider=provider_id,
                                    resolved_model=target_model,
                                    status="failed",
                                    http_status=500,
                                    prompt_tokens=0,
                                    completion_tokens=token_count,
                                    latency_ms=(time.time() - start_t) * 1000,
                                    tier=tier,
                                    stream=True,
                                    candidates_attempted=candidates_attempted,
                                    error_message=f"Mid-stream disconnection: {stream_err}",
                                )
                                raise stream_err

                        candidates_attempted.append({
                            "provider": provider_id,
                            "model": target_model,
                            "key": key_tag,
                            "status": "success",
                        })

                        route_metadata = {
                            "tier": tier,
                            "classification_method": method,
                            "classification_ms": classify_ms,
                            "candidates_attempted": candidates_attempted,
                            "preemptive_bypasses": preemptive_bypasses,
                        }
                        return tracked_generator(), provider_id, target_model, route_metadata

                    except Exception as exc:
                        latency_ms = (time.time() - start_t) * 1000
                        should_cooldown, cd_sec, err_status, err_detail = self._evaluate_error(
                            exc, provider_id, target_model
                        )
                        if should_cooldown:
                            key_state.mark_cooldown(cd_sec, err_detail)
                        candidates_attempted.append({
                            "provider": provider_id,
                            "model": target_model,
                            "key": key_tag,
                            "status": err_status,
                            "detail": err_detail,
                            "cooldown_applied": should_cooldown,
                            "cooldown_seconds": cd_sec if should_cooldown else 0,
                        })
                        logger.warning(
                            "Stream candidate failed %s/%s with key %s (%s, cooldown=%s): %s. Checking next available key before model failover...",
                            provider_id,
                            target_model,
                            key_tag,
                            err_status,
                            should_cooldown,
                            err_detail,
                        )
                        if err_status in ("model_unavailable", "invalid_request", "context_exceeded", "subscription_required", "execution_refused"):
                            logger.info(
                                "Model-level failure '%s' on %s/%s. Skipping remaining keys for this model and advancing cascade.",
                                err_status,
                                provider_id,
                                target_model,
                            )
                            break

        total_latency_ms = (time.time() - req_start_t) * 1000
        error_summary = "; ".join([f"{c['provider']}: {c['status']} ({c.get('detail', '')[:80]})" for c in candidates_attempted])

        history_manager.record_request(
            endpoint=endpoint,
            protocol=protocol,
            requested_model=requested_model,
            resolved_provider=None,
            resolved_model=None,
            status="failed",
            http_status=503,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=total_latency_ms,
            tier=tier,
            stream=True,
            candidates_attempted=candidates_attempted,
            error_message=f"All fallback stream candidates exhausted: {error_summary}",
        )

        raise ProviderException(
            f"All fallback stream candidates exhausted for model '{requested_model}' in tier '{tier}'. Summary: {error_summary}",
            status_code=503,
            diagnostics={
                "tier": tier,
                "classification_method": method,
                "classification_ms": classify_ms,
                "candidates_attempted": candidates_attempted,
                "preemptive_bypasses": preemptive_bypasses,
            },
        )


    async def shutdown(self):
        """Cleanly close all open provider HTTP clients."""
        for p_id, provider in self.providers.items():
            if hasattr(provider, "client") and hasattr(provider.client, "aclose"):
                try:
                    await provider.client.aclose()
                except Exception as e:
                    logger.debug("Error closing HTTP client for provider %s: %s", p_id, e)


router_engine = RoutingEngine()
