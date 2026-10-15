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

    def resolve_fallback_chain(self, requested_model: str, tier: str = "CODING") -> List[Tuple[str, str]]:
        """
        Resolves model and reorders candidate chain based on the detected tier.
        Respects active routing strategy and disabled model toggles.
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
        alias_cfg = config.models.get(requested_model) or config.models.get(req_clean)
        if not alias_cfg:
            if "haiku" in req_clean or "fast" in req_clean or "plan" in req_clean or "mini" in req_clean:
                alias_cfg = config.models.get("fast-planning-light") or config.models.get("fast")
            elif "3-5-sonnet" in req_clean or "3.5-sonnet" in req_clean or "gpt-4" in req_clean or "gpt-5" in req_clean or "general" in req_clean or "chat" in req_clean or "luna" in req_clean:
                alias_cfg = config.models.get("general-massive-quota") or config.models.get("general")
            elif "sonnet" in req_clean or "opus" in req_clean or "code" in req_clean or "reason" in req_clean or "r1" in req_clean:
                alias_cfg = config.models.get("pro-coding-reasoning") or config.models.get("coding")

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
                default_cfg = config.models.get("pro-coding-reasoning") or config.models.get("claude-3-7-sonnet-latest")
                base_chain = [(step.provider, step.model) for step in default_cfg.fallbacks] if default_cfg else [("gemini", "gemini-3.8-flash")]

        if not base_chain:
            base_chain = [("gemini", "gemini-3.8-flash"), ("cerebras", "gpt-oss-120b")]

        # Virtual stacks and curated pools should run exactly as specified
        curated_stacks = (
            "pro-coding-reasoning", "fast-planning-light", "general-massive-quota",
            "claude-3-7-sonnet-latest", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
            "reasoning", "coding", "general", "fast", "vision", "quality-pool", "mega-pool"
        )
        if (requested_model in curated_stacks or req_clean in curated_stacks) and not prefix_primary:
            return self._filter_disabled(base_chain)

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
                return self._filter_disabled(ordered_chain)
            # If strategy is 'direct' (empty priority), use base chain as-is
            if active_strategy == "direct":
                if prefix_primary:
                    base_chain = [prefix_primary] + [x for x in base_chain if x != prefix_primary]
                return self._filter_disabled(base_chain)

        # 4. Dynamic Tier-Based Re-Ordering & Prioritization (default strategy)
        if tier == "CASUAL":
            casual_pool = [
                ("apinex", "free/gemini-3.8-flash"),
                ("apinex", "free/gpt-6-luna"),
                ("apinex", "free/deepseek-v4.1-flash"),
                ("gemini", "gemini-3.8-flash"),      # 1,500 RPD, 15 RPM
                ("gemini", "gemini-3.5-flash-lite"),  # 500 RPD, 15 RPM
                ("gemini", "gemini-3.1-flash-lite"),  # 500 RPD, 15 RPM
                ("gemini", "gemma-4-31b-it"),        # 14,000 RPD, 30 RPM
                ("groq", "openai/gpt-oss-20b"),      # 14,400 RPD, 300 tok/s fast free
                ("cerebras", "qwen-3.8-27b"),        # 14,400 RPD, 2,000 tok/s ultra-fast
                ("cohere", "command-r-08-2024"),     # 1,000 RPD
            ]

            openrouter_avail, _ = rate_budget.check_availability("openrouter")
            openrouter_models = [
                ("openrouter", "qwen/qwen3.8-27b:free"),
                ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
            ]

            ordered_chain = []
            ordered_chain.extend(casual_pool)
            if openrouter_avail:
                ordered_chain.extend(openrouter_models)

            # Append the rest of base chain as safety net
            for item in base_chain:
                if item not in ordered_chain:
                    ordered_chain.append(item)

        elif tier == "CODING":
            coding_priority = [
                ("apinex", "free/qwen-3.8-max"),
                ("apinex", "free/deepseek-v4-pro-0813"),
                ("apinex", "free/deepseek-v4.1-flash"),
                ("antigravity", "gemini-2.5-pro"),
                ("mistral", "codestral-latest"),
                ("cerebras", "gpt-oss-120b"),
                ("groq", "openai/gpt-oss-120b"),
                ("gemini", "gemini-3.8-flash"),
                ("huggingface", "Qwen/Qwen2.5-Coder-32B-Instruct"),
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
                ("apinex", "free/qwen-3.8-max"),
                ("apinex", "free/deepseek-v4-pro-0813"),
                ("antigravity", "gemini-2.5-pro"),
                ("deepseek", "deepseek-reasoner"),
                ("cerebras", "gpt-oss-120b"),
                ("gemini", "gemini-3.8-flash"),
                ("groq", "openai/gpt-oss-120b"),
                ("mistral", "codestral-latest"),
                ("cohere", "command-r-plus-08-2024"),
                ("nvidia", "deepseek-ai/deepseek-v4.1-flash"),
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

        return self._filter_disabled(ordered_chain)

    async def execute_chat(
        self,
        requested_model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Tuple[UnifiedChatResult, str, str, Dict[str, Any]]:
        """
        Executes non-streaming completion with intelligent tier classification and pre-emptive rate budgeting.
        Returns: (UnifiedChatResult, resolved_provider, resolved_model, route_metadata)
        """
        # 1. Classify prompt
        tier, method, classify_ms = await classifier.classify(
            messages,
            micro_infer_func=self._micro_classify,
        )

        chain = self.resolve_fallback_chain(requested_model, tier=tier)
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

                key_state = key_manager.get_key(provider_id)
                if not key_state:
                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
                        "status": "no_key",
                        "detail": f"Provider '{provider_id}' has no available active keys or is cooling down",
                    })
                    continue

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
                        "status": "soft_throttled",
                        "detail": budget_reason,
                    })
                    continue

                start_t = time.time()
                try:
                    logger.info(
                        "[%s] Routing '%s' -> %s/%s using key ...%s",
                        tier,
                        requested_model,
                        provider_id,
                        target_model,
                        key_state.key[-6:] if len(key_state.key) >= 6 else "key",
                    )
                    result = await provider.chat_completion(
                        model=target_model,
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_key=key_state.key,
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
                        "status": "success",
                        "latency_ms": round(latency_ms, 1),
                    })

                    route_metadata = {
                        "tier": tier,
                        "classification_method": method,
                        "classification_ms": classify_ms,
                        "candidates_attempted": candidates_attempted,
                        "preemptive_bypasses": preemptive_bypasses,
                    }
                    return result, provider_id, target_model, route_metadata

                except RateLimitException as rle:
                    latency_ms = (time.time() - start_t) * 1000
                    key_state.mark_cooldown(
                        duration_seconds=rle.retry_after,
                        reason=f"HTTP 429: {rle.message}",
                    )
                    rate_tracker.record_request(
                        provider_id=provider_id,
                        model_id=target_model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        success=False,
                        is_429=True,
                    )
                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
                        "status": "rate_limited",
                        "detail": f"HTTP 429: {rle.message}",
                        "cooldown_seconds": rle.retry_after,
                    })
                    logger.warning("Rate limit on %s/%s. Auto-failing over...", provider_id, target_model)

                except Exception as exc:
                    key_state.mark_cooldown(30.0, str(exc))
                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
                        "status": "error",
                        "detail": str(exc),
                    })
                    logger.warning("Error on %s/%s: %s. Auto-failing over...", provider_id, target_model, exc)

        # If all candidates in chain failed
        error_summary = "; ".join([f"{c['provider']}: {c['status']} ({c.get('detail', '')[:80]})" for c in candidates_attempted])
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
    ) -> Tuple[AsyncGenerator[Dict[str, Any], None], str, str, Dict[str, Any]]:
        """
        Executes streaming completion with intelligent tier classification and pre-emptive rate budgeting.
        Returns: (chunk_generator, resolved_provider, resolved_model, route_metadata)
        """
        # 1. Classify prompt
        tier, method, classify_ms = await classifier.classify(
            messages,
            micro_infer_func=self._micro_classify,
        )

        chain = self.resolve_fallback_chain(requested_model, tier=tier)
        candidates_attempted: List[Dict[str, Any]] = []
        preemptive_bypasses: List[Dict[str, Any]] = []

        for allow_soft in [False, True]:
            for provider_id, target_model in chain:
                if any(c["provider"] == provider_id and c["model"] == target_model and c["status"] != "soft_throttled" for c in candidates_attempted):
                    continue

                provider = self.providers.get(provider_id)
                if not provider:
                    continue

                key_state = key_manager.get_key(provider_id)
                if not key_state:
                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
                        "status": "no_key",
                        "detail": f"Provider '{provider_id}' has no available active keys or is cooling down",
                    })
                    continue

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
                        "status": "soft_throttled",
                        "detail": budget_reason,
                    })
                    continue

                start_t = time.time()
                try:
                    logger.info(
                        "[%s] Streaming route '%s' -> %s/%s using key ...%s",
                        tier,
                        requested_model,
                        provider_id,
                        target_model,
                        key_state.key[-6:] if len(key_state.key) >= 6 else "key",
                    )
                    stream_gen = provider.stream_chat_completion(
                        model=target_model,
                        messages=messages,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_key=key_state.key,
                    )

                    # Prime the generator to ensure stream connects successfully
                    stream_iter = stream_gen.__aiter__()
                    try:
                        first_chunk = await stream_iter.__anext__()
                    except StopAsyncIteration:
                        first_chunk = None

                    rate_budget.record_request(provider_id, model=target_model)

                    async def tracked_generator():
                        token_count = 0
                        try:
                            if first_chunk is not None:
                                if first_chunk.get("text"):
                                    token_count += 1
                                yield first_chunk

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
                        except Exception as stream_err:
                            key_state.mark_cooldown(30.0, str(stream_err))
                            logger.error("Mid-stream error on %s/%s: %s", provider_id, target_model, stream_err)
                            raise stream_err

                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
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

                except RateLimitException as rle:
                    key_state.mark_cooldown(rle.retry_after, f"429: {rle.message}")
                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
                        "status": "rate_limited",
                        "detail": f"HTTP 429: {rle.message}",
                        "cooldown_seconds": rle.retry_after,
                    })
                    logger.warning("Stream rate limit on %s/%s. Auto-failing over...", provider_id, target_model)

                except Exception as exc:
                    key_state.mark_cooldown(30.0, f"Error: {exc}")
                    candidates_attempted.append({
                        "provider": provider_id,
                        "model": target_model,
                        "status": "error",
                        "detail": str(exc),
                    })
                    logger.warning("Stream error on %s/%s (%s). Auto-failing over...", provider_id, target_model, exc)

        error_summary = "; ".join([f"{c['provider']}: {c['status']} ({c.get('detail', '')[:80]})" for c in candidates_attempted])
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
