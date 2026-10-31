"""Real-time metrics, token tracking, and rate monitoring."""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Tuple


class RequestRecord:
    def __init__(self, timestamp: float, tokens: int, latency_ms: float, success: bool):
        self.timestamp = timestamp
        self.tokens = tokens
        self.latency_ms = latency_ms
        self.success = success


class ModelUsageRecord:
    def __init__(self, model_id: str, provider_id: str):
        self.model_id = model_id
        self.provider_id = provider_id
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.latencies: collections.deque[float] = collections.deque(maxlen=100)
        self.last_used: float = 0.0

    def record(self, p_tok: int, c_tok: int, lat_ms: float, success: bool):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += (p_tok + c_tok)
        self.latencies.append(lat_ms)
        self.last_used = time.time()

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return round(sum(self.latencies) / len(self.latencies), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "last_used": self.last_used,
        }


class ProviderMetrics:
    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        # History queue for recent 60-second and 24-hour calculations
        self.history: collections.deque[RequestRecord] = collections.deque(maxlen=5000)

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        is_429: bool = False,
        timestamp: Optional[float] = None,
    ):
        now = timestamp or time.time()
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if is_429:
            self.rate_limited_count += 1

        total = max(0, prompt_tokens) + max(0, completion_tokens)
        self.total_prompt_tokens += max(0, prompt_tokens)
        self.total_completion_tokens += max(0, completion_tokens)
        self.total_tokens += total

        self.history.append(
            RequestRecord(
                timestamp=now,
                tokens=total,
                latency_ms=latency_ms,
                success=success,
            )
        )

    def get_current_rpm_tpm(self) -> tuple[int, int]:
        """Calculate requests and tokens within the last 60 seconds."""
        now = time.time()
        cutoff = now - 60.0
        rpm = 0
        tpm = 0
        for rec in reversed(self.history):
            if rec.timestamp < cutoff:
                break
            rpm += 1
            tpm += rec.tokens
        return rpm, tpm

    def get_24h_requests_tokens(self) -> tuple[int, int]:
        """Calculate requests and tokens within the last 24 hours."""
        now = time.time()
        cutoff = now - 86400.0
        rpd = 0
        tpd = 0
        for rec in reversed(self.history):
            if rec.timestamp < cutoff:
                break
            rpd += 1
            tpd += rec.tokens
        return rpd, tpd

    def get_avg_latency_ms(self) -> float:
        if not self.history:
            return 0.0
        recent = list(self.history)[-50:]
        latencies = [r.latency_ms for r in recent if r.success and r.latency_ms > 0]
        return round(sum(latencies) / len(latencies), 1) if latencies else 0.0


class RateTracker:
    def __init__(self):
        self.providers: Dict[str, ProviderMetrics] = {}
        self.model_requests: collections.Counter = collections.Counter()
        self.model_metrics: Dict[str, ModelUsageRecord] = {}
        self.start_time = time.time()
        self._seeded = False

    def get_provider_metrics(self, provider_id: str) -> ProviderMetrics:
        if provider_id not in self.providers:
            self.providers[provider_id] = ProviderMetrics(provider_id)
        return self.providers[provider_id]

    def get_model_metric(self, provider_id: str, model_id: str) -> ModelUsageRecord:
        key = f"{provider_id}:{model_id}"
        if key not in self.model_metrics:
            self.model_metrics[key] = ModelUsageRecord(model_id, provider_id)
        return self.model_metrics[key]

    def record_request(
        self,
        provider_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        is_429: bool = False,
        timestamp: Optional[float] = None,
    ):
        metrics = self.get_provider_metrics(provider_id)
        metrics.record(prompt_tokens, completion_tokens, latency_ms, success, is_429, timestamp=timestamp)
        self.model_requests[model_id] += 1

        m_metric = self.get_model_metric(provider_id, model_id)
        m_metric.record(prompt_tokens, completion_tokens, latency_ms, success)

    def seed_from_history(self, history_records: List[Dict[str, Any]]):
        """Seed recent request counts from persistent history.json."""
        if self._seeded or not history_records:
            return
        now = time.time()
        cutoff_24h = now - 86400.0
        for rec in history_records:
            t = rec.get("epoch") or 0.0
            if t < cutoff_24h:
                continue
            prov = rec.get("resolved_provider")
            mod = rec.get("resolved_model")
            if not prov or prov in ("none", "unknown"):
                continue
            p_tok = rec.get("prompt_tokens", 0)
            c_tok = rec.get("completion_tokens", 0)
            lat = rec.get("latency_ms", 0.0)
            succ = rec.get("status") == "fulfilled"
            is_429 = "429" in str(rec.get("error_message") or "")
            self.record_request(
                provider_id=prov,
                model_id=mod or "default",
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat,
                success=succ,
                is_429=is_429,
                timestamp=t,
            )
        self._seeded = True

    def get_overview(self) -> dict:
        total_reqs = sum(p.total_requests for p in self.providers.values())
        total_success = sum(p.successful_requests for p in self.providers.values())
        total_tokens = sum(p.total_tokens for p in self.providers.values())
        total_rpm = 0
        total_tpm = 0

        prov_data = {}
        for p_id, p_metrics in self.providers.items():
            rpm, tpm = p_metrics.get_current_rpm_tpm()
            rpd, tpd = p_metrics.get_24h_requests_tokens()
            total_rpm += rpm
            total_tpm += tpm
            prov_data[p_id] = {
                "total_requests": p_metrics.total_requests,
                "successful_requests": p_metrics.successful_requests,
                "failed_requests": p_metrics.failed_requests,
                "rate_limited_count": p_metrics.rate_limited_count,
                "current_rpm": rpm,
                "current_tpm": tpm,
                "today_requests": rpd,
                "today_tokens": tpd,
                "total_tokens": p_metrics.total_tokens,
                "avg_latency_ms": p_metrics.get_avg_latency_ms(),
                "success_rate": round(
                    (p_metrics.successful_requests / p_metrics.total_requests * 100)
                    if p_metrics.total_requests > 0
                    else 100.0,
                    1,
                ),
            }

        return {
            "uptime_seconds": int(time.time() - self.start_time),
            "total_requests": total_reqs,
            "successful_requests": total_success,
            "overall_success_rate": round(
                (total_success / total_reqs * 100) if total_reqs > 0 else 100.0, 1
            ),
            "total_tokens": total_tokens,
            "current_rpm": total_rpm,
            "current_tpm": total_tpm,
            "providers": prov_data,
            "top_models": dict(self.model_requests.most_common(10)),
            "model_metrics": {k: v.to_dict() for k, v in self.model_metrics.items()},
        }

    def get_comprehensive_usage(self) -> Dict[str, Any]:
        """
        Aggregate live usage metrics, pool status, and limits for all 39 providers
        and all associated models.
        """
        from app.core.config import config
        from app.core.key_pool import key_manager
        from app.core.state_manager import state_manager

        # Lazy seed from persistent history if not done
        if not self._seeded:
            try:
                from app.core.history_manager import history_manager
                self.seed_from_history(history_manager.get_history(limit=500).get("records", []))
            except Exception:
                pass

        now = time.time()
        disabled_map = state_manager.get_disabled_models()
        health_cache, _ = state_manager.get_health_cache()

        providers_list = []
        total_pooled_keys = 0
        active_keys_count = 0
        cooling_keys_count = 0
        total_requests_all = 0
        total_tokens_all = 0
        today_requests_all = 0
        current_rpm_all = 0
        current_tpm_all = 0
        throttled_providers_count = 0

        for p_id, p_def in config.providers.items():
            pool = key_manager.get_pool(p_id)
            has_keys = pool is not None and pool.total_keys > 0
            key_count = pool.total_keys if pool else 0
            total_pooled_keys += key_count

            # Evaluate key states
            key_details = []
            keys_available = 0
            keys_cooling = 0
            max_cooldown_remaining = 0.0

            if pool and pool.keys:
                for idx, k_state in enumerate(pool.keys):
                    cd_remaining = max(0.0, round(k_state.cooldown_until - now, 1))
                    is_avail = k_state.is_available
                    if is_avail:
                        keys_available += 1
                        active_keys_count += 1
                    else:
                        keys_cooling += 1
                        cooling_keys_count += 1
                        if cd_remaining > max_cooldown_remaining:
                            max_cooldown_remaining = cd_remaining

                    masked = k_state.key[:6] + "..." + k_state.key[-4:] if len(k_state.key) >= 12 else "key-" + str(idx + 1)
                    key_details.append({
                        "index": idx + 1,
                        "masked_key": masked,
                        "is_available": is_avail,
                        "cooldown_remaining_sec": cd_remaining,
                        "consecutive_errors": k_state.consecutive_errors,
                        "total_requests": k_state.total_requests,
                        "total_failures": k_state.total_failures,
                        "last_used_seconds_ago": round(now - k_state.last_used, 1) if k_state.last_used > 0 else None,
                    })

            # Usage stats
            p_metrics = self.providers.get(p_id)
            rpm = 0
            tpm = 0
            today_reqs = 0
            today_toks = 0
            total_reqs = 0
            total_toks = 0
            failed_reqs = 0
            rate_limited_count = 0
            avg_lat = 0.0

            if p_metrics:
                rpm, tpm = p_metrics.get_current_rpm_tpm()
                today_reqs, today_toks = p_metrics.get_24h_requests_tokens()
                total_reqs = p_metrics.total_requests
                total_toks = p_metrics.total_tokens
                failed_reqs = p_metrics.failed_requests
                rate_limited_count = p_metrics.rate_limited_count
                avg_lat = p_metrics.get_avg_latency_ms()

            total_requests_all += total_reqs
            total_tokens_all += total_toks
            today_requests_all += today_reqs
            current_rpm_all += rpm
            current_tpm_all += tpm

            # Limits from providers.yaml
            limit_rpm = p_def.free_tier.rpm or 20
            limit_rpd = p_def.free_tier.rpd or 1000
            limit_tpm = p_def.free_tier.tpm or 100000

            rpm_pct = round(min(100.0, (rpm / limit_rpm * 100.0)), 1) if limit_rpm > 0 else 0.0
            rpd_pct = round(min(100.0, (today_reqs / limit_rpd * 100.0)), 1) if limit_rpd > 0 else 0.0
            tpm_pct = round(min(100.0, (tpm / limit_tpm * 100.0)), 1) if limit_tpm > 0 else 0.0

            # Determine provider status
            if not has_keys:
                status = "no_keys"
                status_label = "No Keys"
            elif keys_cooling > 0 and keys_available == 0:
                status = "cooling_down"
                status_label = f"Cooling Down ({int(max_cooldown_remaining)}s)"
                throttled_providers_count += 1
            elif rpd_pct >= 100.0:
                status = "exhausted"
                status_label = "Daily Limit Reached"
                throttled_providers_count += 1
            elif rpm_pct >= 90.0 or rpd_pct >= 85.0:
                status = "near_limit"
                status_label = "High Load"
            elif keys_cooling > 0 and keys_available > 0:
                status = "degraded"
                status_label = f"Degraded ({keys_cooling} cooling)"
            else:
                status = "active"
                status_label = "Active & Ready"

            # Model breakdown under this provider
            models_data = []
            for m in p_def.models:
                m_key = f"{p_id}:{m.id}"
                m_metric = self.model_metrics.get(m_key)
                health = health_cache.get(m_key, {})
                models_data.append({
                    "id": m.id,
                    "name": m.name,
                    "context_window": m.context_window,
                    "supports_tools": m.supports_tools,
                    "supports_vision": m.supports_vision,
                    "supports_thinking": m.supports_thinking,
                    "disabled": disabled_map.get(m_key, False),
                    "health_status": health.get("status", "unknown"),
                    "total_requests": m_metric.total_requests if m_metric else 0,
                    "total_tokens": m_metric.total_tokens if m_metric else 0,
                    "avg_latency_ms": m_metric.avg_latency_ms if m_metric else 0.0,
                    "last_used_seconds_ago": (
                        round(now - m_metric.last_used, 1) if (m_metric and m_metric.last_used > 0) else None
                    ),
                })

            providers_list.append({
                "id": p_id,
                "name": p_def.name,
                "type": p_def.type,
                "docs_url": p_def.docs_url,
                "has_keys": has_keys,
                "key_count": key_count,
                "keys_available": keys_available,
                "keys_cooling": keys_cooling,
                "max_cooldown_sec": max_cooldown_remaining,
                "key_details": key_details,
                "status": status,
                "status_label": status_label,
                "limits": {
                    "rpm": limit_rpm,
                    "rpd": limit_rpd,
                    "tpm": limit_tpm,
                },
                "usage": {
                    "current_rpm": rpm,
                    "current_tpm": tpm,
                    "today_requests": today_reqs,
                    "today_tokens": today_toks,
                    "total_requests": total_reqs,
                    "total_tokens": total_toks,
                    "failed_requests": failed_reqs,
                    "rate_limited_count": rate_limited_count,
                    "avg_latency_ms": avg_lat,
                },
                "utilization_pct": {
                    "rpm": rpm_pct,
                    "rpd": rpd_pct,
                    "tpm": tpm_pct,
                },
                "models": models_data,
            })

        # Sort: active providers with keys first, then alphabetically
        providers_list.sort(key=lambda x: (not x["has_keys"], x["status"] == "no_keys", x["name"].lower()))

        # Top 5 most used providers & models
        top_providers = sorted(
            [p for p in providers_list if p["usage"]["total_requests"] > 0],
            key=lambda x: x["usage"]["total_requests"],
            reverse=True,
        )[:6]

        top_models = sorted(
            [
                {
                    "model_id": k.split(":", 1)[1] if ":" in k else k,
                    "provider_id": k.split(":", 1)[0] if ":" in k else "unknown",
                    "total_requests": v.total_requests,
                    "total_tokens": v.total_tokens,
                    "avg_latency_ms": v.avg_latency_ms,
                }
                for k, v in self.model_metrics.items()
                if v.total_requests > 0
            ],
            key=lambda x: x["total_requests"],
            reverse=True,
        )[:8]

        return {
            "summary": {
                "uptime_seconds": int(now - self.start_time),
                "total_providers": len(config.providers),
                "active_providers_with_keys": sum(1 for p in providers_list if p["has_keys"]),
                "cooling_down_providers": throttled_providers_count,
                "total_pooled_keys": total_pooled_keys,
                "active_keys_count": active_keys_count,
                "cooling_keys_count": cooling_keys_count,
                "today_requests": today_requests_all,
                "total_requests_all_time": total_requests_all,
                "total_tokens_all_time": total_tokens_all,
                "current_throughput_rpm": current_rpm_all,
                "current_throughput_tpm": current_tpm_all,
            },
            "top_providers": top_providers,
            "top_models": top_models,
            "providers": providers_list,
        }

    def reset_metrics(self):
        """Reset live counters."""
        self.providers.clear()
        self.model_requests.clear()
        self.model_metrics.clear()
        self.start_time = time.time()
        self._seeded = True


rate_tracker = RateTracker()

