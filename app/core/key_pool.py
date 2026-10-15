"""Multi-key round-robin pool with cooldown management for free LLM providers."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KeyState:
    def __init__(self, key: str):
        self.key = key
        self.cooldown_until: float = 0.0
        self.consecutive_errors: int = 0
        self.total_requests: int = 0
        self.total_failures: int = 0
        self.last_used: float = 0.0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def available(self) -> bool:
        return self.is_available

    def mark_cooldown(self, duration_seconds: float, reason: str = "Rate limited"):
        self.cooldown_until = time.time() + duration_seconds
        self.consecutive_errors += 1
        self.total_failures += 1
        logger.warning(
            "Key ...%s in cooldown for %.1fs (reason: %s). Consecutive errors: %d",
            self.key[-6:] if len(self.key) >= 6 else self.key,
            duration_seconds,
            reason,
            self.consecutive_errors,
        )

    def mark_success(self):
        self.consecutive_errors = 0
        self.total_requests += 1
        self.last_used = time.time()


class ProviderKeyPool:
    def __init__(self, provider_id: str, keys: List[str]):
        self.provider_id = provider_id
        self.keys = [KeyState(k) for k in keys]
        self._index = 0

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    @property
    def available_keys_count(self) -> int:
        return sum(1 for k in self.keys if k.is_available)

    def get_next_key(self) -> Optional[KeyState]:
        """Round-robin select the next available key not in cooldown."""
        if not self.keys:
            return None

        now = time.time()
        # Scan from current index
        for _ in range(len(self.keys)):
            k = self.keys[self._index % len(self.keys)]
            self._index = (self._index + 1) % len(self.keys)
            if k.is_available:
                return k

        # If all keys are in cooldown, find the key that will become available soonest
        soonest = min(self.keys, key=lambda k: k.cooldown_until)
        wait_time = max(0.0, soonest.cooldown_until - now)
        logger.warning(
            "Provider '%s': All %d keys in cooldown. Soonest available in %.1fs.",
            self.provider_id,
            len(self.keys),
            wait_time,
        )
        return None

    def update_keys(self, new_keys: List[str]):
        """Dynamically update key list while preserving active cooldown states."""
        existing = {k.key: k for k in self.keys}
        updated: List[KeyState] = []
        for raw in new_keys:
            if raw in existing:
                updated.append(existing[raw])
            else:
                updated.append(KeyState(raw))
        self.keys = updated


class KeyPoolManager:
    """Manages key pools across all registered providers."""

    def __init__(self):
        self.pools: Dict[str, ProviderKeyPool] = {}

    def register_provider(self, provider_id: str, keys: List[str]):
        self.pools[provider_id] = ProviderKeyPool(provider_id, keys)

    def get_key(self, provider_id: str) -> Optional[KeyState]:
        pool = self.pools.get(provider_id)
        if not pool:
            return None
        return pool.get_next_key()

    def get_pool(self, provider_id: str) -> Optional[ProviderKeyPool]:
        return self.pools.get(provider_id)

    def get_status(self) -> Dict[str, dict]:
        now = time.time()
        result = {}
        for p_id, pool in self.pools.items():
            result[p_id] = {
                "total_keys": pool.total_keys,
                "available_keys": pool.available_keys_count,
                "keys": [
                    {
                        "masked_key": f"...{k.key[-6:]}" if len(k.key) >= 6 else "empty",
                        "available": k.is_available,
                        "cooldown_remaining": max(0.0, round(k.cooldown_until - now, 1)),
                        "total_requests": k.total_requests,
                        "total_failures": k.total_failures,
                    }
                    for k in pool.keys
                ],
            }
        return result


# Global singleton key pool
key_manager = KeyPoolManager()
