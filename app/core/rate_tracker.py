"""Real-time metrics, token tracking, and rate monitoring."""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional


class RequestRecord:
    def __init__(self, timestamp: float, tokens: int, latency_ms: float, success: bool):
        self.timestamp = timestamp
        self.tokens = tokens
        self.latency_ms = latency_ms
        self.success = success


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
        self.history: collections.deque[RequestRecord] = collections.deque(maxlen=1000)

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        is_429: bool = False,
    ):
        now = time.time()
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        if is_429:
            self.rate_limited_count += 1

        total = prompt_tokens + completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
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

    def get_avg_latency_ms(self) -> float:
        if not self.history:
            return 0.0
        recent = list(self.history)[-50:]
        latencies = [r.latency_ms for r in recent if r.success]
        return round(sum(latencies) / len(latencies), 1) if latencies else 0.0


class RateTracker:
    def __init__(self):
        self.providers: Dict[str, ProviderMetrics] = {}
        self.model_requests: collections.Counter = collections.Counter()
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()

    def get_provider_metrics(self, provider_id: str) -> ProviderMetrics:
        if provider_id not in self.providers:
            self.providers[provider_id] = ProviderMetrics(provider_id)
        return self.providers[provider_id]

    def record_request(
        self,
        provider_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool,
        is_429: bool = False,
    ):
        metrics = self.get_provider_metrics(provider_id)
        metrics.record(prompt_tokens, completion_tokens, latency_ms, success, is_429)
        self.model_requests[model_id] += 1

        m_key = f"{provider_id}:{model_id}"
        if m_key not in self.model_metrics:
            self.model_metrics[m_key] = {"total_requests": 0, "total_tokens": 0, "last_used": 0.0}
        self.model_metrics[m_key]["total_requests"] += 1
        self.model_metrics[m_key]["total_tokens"] += (prompt_tokens + completion_tokens)
        self.model_metrics[m_key]["last_used"] = time.time()

    def get_overview(self) -> dict:
        total_reqs = sum(p.total_requests for p in self.providers.values())
        total_success = sum(p.successful_requests for p in self.providers.values())
        total_tokens = sum(p.total_tokens for p in self.providers.values())
        total_rpm = 0
        total_tpm = 0

        prov_data = {}
        for p_id, p_metrics in self.providers.items():
            rpm, tpm = p_metrics.get_current_rpm_tpm()
            total_rpm += rpm
            total_tpm += tpm
            prov_data[p_id] = {
                "total_requests": p_metrics.total_requests,
                "successful_requests": p_metrics.successful_requests,
                "failed_requests": p_metrics.failed_requests,
                "rate_limited_count": p_metrics.rate_limited_count,
                "current_rpm": rpm,
                "current_tpm": tpm,
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
            "model_metrics": self.model_metrics,
        }


rate_tracker = RateTracker()
