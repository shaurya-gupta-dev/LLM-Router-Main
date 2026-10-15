"""Predictive sliding-window rate budgeter and soft-throttler for LLM Router."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProviderBudgetLimits:
    rpm_limit: int = 20
    rpd_limit: int = 1000
    soft_threshold_pct: float = 0.75  # Rotate when reaching 75% of limit


class ProviderBudgetTracker:
    def __init__(self, provider_id: str, limits: ProviderBudgetLimits):
        self.provider_id = provider_id
        self.limits = limits
        self.timestamps_60s: Deque[float] = deque()
        self.timestamps_24h: Deque[float] = deque()

    def _purge_expired(self, now: float):
        cutoff_60s = now - 60.0
        while self.timestamps_60s and self.timestamps_60s[0] < cutoff_60s:
            self.timestamps_60s.popleft()

        cutoff_24h = now - 86400.0
        while self.timestamps_24h and self.timestamps_24h[0] < cutoff_24h:
            self.timestamps_24h.popleft()

    def record_request(self, now: Optional[float] = None):
        t = now or time.time()
        self._purge_expired(t)
        self.timestamps_60s.append(t)
        self.timestamps_24h.append(t)

    def current_usage(self, now: Optional[float] = None) -> Tuple[int, int]:
        t = now or time.time()
        self._purge_expired(t)
        return len(self.timestamps_60s), len(self.timestamps_24h)

    def is_available(
        self,
        now: Optional[float] = None,
        allow_soft_throttled: bool = False,
    ) -> Tuple[bool, str]:
        t = now or time.time()
        self._purge_expired(t)

        rpm_count = len(self.timestamps_60s)
        rpd_count = len(self.timestamps_24h)

        # 1. Hard daily limit check
        if rpd_count >= self.limits.rpd_limit:
            return False, f"Daily limit reached ({rpd_count}/{self.limits.rpd_limit} RPD)"

        # 2. Hard per-minute limit check
        if rpm_count >= self.limits.rpm_limit:
            oldest = self.timestamps_60s[0] if self.timestamps_60s else t
            wait_s = max(1.0, round(60.0 - (t - oldest), 1))
            return False, f"RPM limit reached ({rpm_count}/{self.limits.rpm_limit}). Cooldown: {wait_s}s"

        # 3. Soft pre-emptive threshold check
        soft_rpm = max(1, int(self.limits.rpm_limit * self.limits.soft_threshold_pct))
        if rpm_count >= soft_rpm and not allow_soft_throttled:
            oldest = self.timestamps_60s[0] if self.timestamps_60s else t
            wait_s = max(1.0, round(60.0 - (t - oldest), 1))
            return (
                False,
                f"Soft budget reached ({rpm_count}/{self.limits.rpm_limit} RPM, safe threshold {soft_rpm}). Pre-emptively resting for {wait_s}s",
            )

        return True, "OK"


class RateBudgetManager:
    """Manages predictive rate budgets across all pooled free providers."""

    # Default calibrated limits for free tiers and model-specific quotas
    DEFAULT_CALIBRATIONS: Dict[str, ProviderBudgetLimits] = {
        "openrouter": ProviderBudgetLimits(rpm_limit=20, rpd_limit=50, soft_threshold_pct=0.70),
        "gemini": ProviderBudgetLimits(rpm_limit=15, rpd_limit=1500, soft_threshold_pct=0.80),
        "gemini:gemini-3.8-flash": ProviderBudgetLimits(rpm_limit=15, rpd_limit=1500, soft_threshold_pct=0.80),
        "gemini:gemini-3.5-flash-lite": ProviderBudgetLimits(rpm_limit=15, rpd_limit=500, soft_threshold_pct=0.80),
        "gemini:gemini-3.1-flash-lite": ProviderBudgetLimits(rpm_limit=15, rpd_limit=500, soft_threshold_pct=0.80),
        "gemini:gemma-4-31b-it": ProviderBudgetLimits(rpm_limit=30, rpd_limit=14000, soft_threshold_pct=0.85),
        "cerebras": ProviderBudgetLimits(rpm_limit=30, rpd_limit=14400, soft_threshold_pct=0.75),
        "groq": ProviderBudgetLimits(rpm_limit=30, rpd_limit=14400, soft_threshold_pct=0.75),
        "groq:openai/gpt-oss-120b": ProviderBudgetLimits(rpm_limit=30, rpd_limit=1000, soft_threshold_pct=0.75),
        "groq:openai/gpt-oss-20b": ProviderBudgetLimits(rpm_limit=30, rpd_limit=14400, soft_threshold_pct=0.75),
        "groq:qwen/qwen3.8-27b": ProviderBudgetLimits(rpm_limit=30, rpd_limit=14400, soft_threshold_pct=0.75),
        "mistral": ProviderBudgetLimits(rpm_limit=30, rpd_limit=1000, soft_threshold_pct=0.75),
        "cohere": ProviderBudgetLimits(rpm_limit=20, rpd_limit=1000, soft_threshold_pct=0.80),
        "github": ProviderBudgetLimits(rpm_limit=15, rpd_limit=150, soft_threshold_pct=0.75),
        "huggingface": ProviderBudgetLimits(rpm_limit=20, rpd_limit=1000, soft_threshold_pct=0.75),
        "sambanova": ProviderBudgetLimits(rpm_limit=30, rpd_limit=1000, soft_threshold_pct=0.75),
        "nvidia": ProviderBudgetLimits(rpm_limit=40, rpd_limit=1000, soft_threshold_pct=0.75),
    }

    def __init__(self):
        self._trackers: Dict[str, ProviderBudgetTracker] = {}
        for p_id, limits in self.DEFAULT_CALIBRATIONS.items():
            self._trackers[p_id] = ProviderBudgetTracker(p_id, limits)

    def register_provider(self, provider_id: str, rpm: int, rpd: int):
        limits = self.DEFAULT_CALIBRATIONS.get(
            provider_id, ProviderBudgetLimits(rpm_limit=rpm or 20, rpd_limit=rpd or 1000)
        )
        if rpm > 0:
            limits.rpm_limit = rpm
        if rpd > 0:
            limits.rpd_limit = rpd
        self._trackers[provider_id] = ProviderBudgetTracker(provider_id, limits)

    def check_availability(
        self,
        provider_id: str,
        model: Optional[str] = None,
        allow_soft_throttled: bool = False,
    ) -> Tuple[bool, str]:
        # If model has specific budget tracker, consult it first
        if model:
            model_key = f"{provider_id}:{model}"
            tracker = self._trackers.get(model_key)
            if tracker:
                return tracker.is_available(allow_soft_throttled=allow_soft_throttled)

        # Fall back to provider-level tracker
        tracker = self._trackers.get(provider_id)
        if not tracker:
            return True, "OK (untracked)"
        return tracker.is_available(allow_soft_throttled=allow_soft_throttled)

    def record_request(self, provider_id: str, model: Optional[str] = None):
        tracker = self._trackers.get(provider_id)
        if tracker:
            tracker.record_request()
        if model:
            model_key = f"{provider_id}:{model}"
            m_tracker = self._trackers.get(model_key)
            if m_tracker:
                m_tracker.record_request()

    def get_status(self) -> Dict[str, Any]:
        result = {}
        now = time.time()
        for p_id, tracker in self._trackers.items():
            rpm, rpd = tracker.current_usage(now)
            avail, reason = tracker.is_available(now)
            result[p_id] = {
                "current_rpm": rpm,
                "rpm_limit": tracker.limits.rpm_limit,
                "current_rpd": rpd,
                "rpd_limit": tracker.limits.rpd_limit,
                "available": avail,
                "reason": reason,
                "utilization_pct": round((rpm / tracker.limits.rpm_limit) * 100, 1),
            }
        return result


rate_budget = RateBudgetManager()
