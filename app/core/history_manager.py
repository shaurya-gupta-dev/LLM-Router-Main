"""Persistent Request History & Audit Engine for LLM Router.

Tracks every incoming request across all endpoints (OpenAI / Anthropic),
recording fulfillment status, resolved provider & model, token usage,
latency, and the exact fallback cascade path taken.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
HISTORY_FILE = Path(os.environ.get("ROUTER_HISTORY_FILE", str(BASE_DIR / "history.json")))
MAX_HISTORY_ENTRIES = 500

# Indian Standard Time (IST, UTC+05:30)
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


class HistoryManager:
    """Thread-safe persistent history manager."""

    def __init__(self, max_entries: int = MAX_HISTORY_ENTRIES):
        self._lock = Lock()
        self.max_entries = max_entries
        self._records: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records = data[-self.max_entries :]
                        # Auto-migrate any legacy UTC or 24-hr timestamps to 12-hour AM/PM Indian Standard Time (IST)
                        migrated = False
                        for r in records:
                            ts = str(r.get("timestamp", ""))
                            epoch = r.get("epoch")
                            if epoch and (not ts or "UTC" in ts or (" AM " not in ts and " PM " not in ts)):
                                r["timestamp"] = datetime.fromtimestamp(epoch, tz=IST).strftime("%Y-%m-%d %I:%M:%S %p IST")
                                migrated = True
                        if migrated:
                            try:
                                with open(HISTORY_FILE, "w", encoding="utf-8") as wf:
                                    json.dump(records, wf, indent=2, default=str)
                            except Exception:
                                pass
                        return records
            except Exception as e:
                logger.warning("Failed to load history.json: %s", e)
        return []

    def _save(self):
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._records[-self.max_entries :], f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save history.json: %s", e)

    def record_request(
        self,
        endpoint: str,
        protocol: str,
        requested_model: str,
        resolved_provider: Optional[str],
        resolved_model: Optional[str],
        status: str,  # "fulfilled" | "failed"
        http_status: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        tier: str = "UNKNOWN",
        stream: bool = False,
        candidates_attempted: Optional[List[Dict[str, Any]]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an API request to the history store in Indian Standard Time (IST)."""
        now_dt = datetime.now(IST)
        record = {
            "id": f"req_{uuid.uuid4().hex[:10]}",
            "timestamp": now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST"),
            "epoch": now_dt.timestamp(),
            "endpoint": endpoint,
            "protocol": protocol,
            "requested_model": requested_model,
            "resolved_provider": resolved_provider or "none",
            "resolved_model": resolved_model or "none",
            "status": status,
            "http_status": http_status,
            "prompt_tokens": max(0, prompt_tokens),
            "completion_tokens": max(0, completion_tokens),
            "total_tokens": max(0, prompt_tokens) + max(0, completion_tokens),
            "latency_ms": round(latency_ms, 1),
            "tier": tier,
            "stream": stream,
            "candidates_attempted": candidates_attempted or [],
            "error_message": error_message,
        }

        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_entries:
                self._records = self._records[-self.max_entries :]
            self._save()

        return record

    def get_history(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get filtered history entries and aggregated performance metrics."""
        with self._lock:
            records = list(self._records)

        # Apply filters
        filtered = records
        if status:
            s_clean = status.lower()
            filtered = [r for r in filtered if r.get("status", "").lower() == s_clean]
        if provider and provider != "all":
            p_clean = provider.lower()
            filtered = [r for r in filtered if r.get("resolved_provider", "").lower() == p_clean]

        # Sort descending by epoch
        filtered.sort(key=lambda x: x.get("epoch", 0), reverse=True)
        slice_records = filtered[:limit]

        # Calculate high-level summary metrics
        total_reqs = len(records)
        fulfilled_count = sum(1 for r in records if r.get("status") == "fulfilled")
        failed_count = sum(1 for r in records if r.get("status") == "failed")
        fulfillment_rate = round((fulfilled_count / total_reqs * 100), 1) if total_reqs > 0 else 100.0

        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
        total_comp_tokens = sum(r.get("completion_tokens", 0) for r in records)
        total_tokens = total_prompt_tokens + total_comp_tokens

        fulfilled_latencies = [r.get("latency_ms", 0) for r in records if r.get("status") == "fulfilled" and r.get("latency_ms", 0) > 0]
        avg_latency = round(sum(fulfilled_latencies) / len(fulfilled_latencies), 1) if fulfilled_latencies else 0.0

        return {
            "records": slice_records,
            "total_count": len(filtered),
            "metrics": {
                "total_requests": total_reqs,
                "fulfilled_count": fulfilled_count,
                "failed_count": failed_count,
                "fulfillment_rate": fulfillment_rate,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_comp_tokens,
                "total_tokens": total_tokens,
                "avg_latency_ms": avg_latency,
            },
        }

    def clear_history(self) -> None:
        """Clear all stored history records."""
        with self._lock:
            self._records = []
            self._save()


# Global history singleton
history_manager = HistoryManager()
