"""Dynamic State Manager for LLM Router.

Persists runtime configuration (model toggles, priority hierarchies, routing
strategy presets) to state.json without requiring server restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATE_FILE = Path(os.environ.get("ROUTER_STATE_FILE", str(BASE_DIR / "state.json")))

# -------------------------------------------------------------------
# Default routing strategy presets
# -------------------------------------------------------------------
DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "max_speed": {
        "name": "Max Speed",
        "description": "Prioritizes fastest inference providers (Cerebras ~2000 tok/s, Groq ~300 tok/s, SambaNova)",
        "priority": [
            ("cerebras", "gpt-oss-120b"),
            ("cerebras", "qwen-3.8-27b"),
            ("groq", "openai/gpt-oss-120b"),
            ("groq", "openai/gpt-oss-20b"),
            ("sambanova", "gpt-oss-120b"),
            ("gemini", "gemini-3.8-flash"),
            ("mistral", "codestral-latest"),
        ],
    },
    "best_quality": {
        "name": "Best Quality",
        "description": "Prioritizes highest capability reasoning and coding models",
        "priority": [
            ("gemini", "gemini-3.8-flash"),
            ("mistral", "codestral-latest"),
            ("cerebras", "gpt-oss-120b"),
            ("groq", "openai/gpt-oss-120b"),
            ("cohere", "command-r-plus-08-2024"),
            ("nvidia", "deepseek-ai/deepseek-v4.1-flash"),
            ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
        ],
    },
    "max_quota": {
        "name": "Max Quota (Survival)",
        "description": "Distributes load across providers with the largest free daily quotas to maximize total usage",
        "priority": [
            ("groq", "openai/gpt-oss-20b"),
            ("groq", "qwen/qwen3.8-27b"),
            ("cerebras", "qwen-3.8-27b"),
            ("cerebras", "gpt-oss-120b"),
            ("gemini", "gemma-4-31b-it"),
            ("gemini", "gemini-3.8-flash"),
            ("gemini", "gemini-3.5-flash-lite"),
            ("gemini", "gemini-3.1-flash-lite"),
            ("sambanova", "gpt-oss-120b"),
            ("cohere", "command-r-08-2024"),
            ("mistral", "mistral-small-latest"),
        ],
    },
    "direct": {
        "name": "Direct / No Strategy",
        "description": "Bypasses heuristic tiering — routes directly to the requested model's fallback chain as configured in config.yaml",
        "priority": [],
    },
}


class StateManager:
    """Thread-safe persistent state manager."""

    def __init__(self):
        self._lock = Lock()
        self._state: Dict[str, Any] = self._load()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "disabled_models": {},
            "active_strategy": "default",
            "custom_priority": [],
            "presets": {
                name: {
                    "name": preset["name"],
                    "description": preset["description"],
                    "priority": [list(p) for p in preset["priority"]],
                }
                for name, preset in DEFAULT_PRESETS.items()
            },
            "model_health_cache": {},
            "last_recheck": 0,
            "version": 2,
        }

    def _load(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Ensure all keys exist (migration)
                default = self._default_state()
                for key in default:
                    if key not in data:
                        data[key] = default[key]
                return data
            except Exception as e:
                logger.warning("Failed to load state.json, using defaults: %s", e)
        return self._default_state()

    def _save(self):
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save state.json: %s", e)

    # ---------------------------------------------------------------
    # Model Enable/Disable
    # ---------------------------------------------------------------
    def is_model_disabled(self, provider_id: str, model_id: str) -> bool:
        key = f"{provider_id}:{model_id}"
        with self._lock:
            return self._state.get("disabled_models", {}).get(key, False)

    def toggle_model(self, provider_id: str, model_id: str, disabled: bool) -> None:
        key = f"{provider_id}:{model_id}"
        with self._lock:
            if "disabled_models" not in self._state:
                self._state["disabled_models"] = {}
            self._state["disabled_models"][key] = disabled
            self._save()
        logger.info("Model %s set to %s", key, "DISABLED" if disabled else "ENABLED")

    def get_disabled_models(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._state.get("disabled_models", {}))

    def set_bulk_disabled(self, disabled_map: Dict[str, bool]) -> None:
        with self._lock:
            self._state["disabled_models"] = disabled_map
            self._save()

    # ---------------------------------------------------------------
    # Routing Strategy
    # ---------------------------------------------------------------
    def get_active_strategy(self) -> str:
        with self._lock:
            return self._state.get("active_strategy", "default")

    def set_active_strategy(self, strategy_name: str) -> None:
        with self._lock:
            self._state["active_strategy"] = strategy_name
            self._save()
        logger.info("Active routing strategy set to: %s", strategy_name)

    def get_strategy_priority(self, strategy_name: str) -> Optional[List[Tuple[str, str]]]:
        """Get priority list for a named strategy. Returns None if strategy is 'default' or not found."""
        if strategy_name == "default":
            return None
        with self._lock:
            presets = self._state.get("presets", {})
            preset = presets.get(strategy_name)
            if preset:
                return [tuple(p) for p in preset.get("priority", [])]
            return None

    def get_active_priority(self) -> Optional[List[Tuple[str, str]]]:
        """Get priority list for the currently active strategy."""
        strategy = self.get_active_strategy()
        if strategy == "custom":
            with self._lock:
                custom = self._state.get("custom_priority", [])
                return [tuple(p) for p in custom] if custom else None
        return self.get_strategy_priority(strategy)

    # ---------------------------------------------------------------
    # Custom Priority
    # ---------------------------------------------------------------
    def set_custom_priority(self, priority: List[List[str]]) -> None:
        with self._lock:
            self._state["custom_priority"] = priority
            self._save()

    def get_custom_priority(self) -> List[List[str]]:
        with self._lock:
            return list(self._state.get("custom_priority", []))

    # ---------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------
    def get_all_presets(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state.get("presets", {}))

    def save_preset(self, name: str, preset_data: Dict[str, Any]) -> None:
        with self._lock:
            if "presets" not in self._state:
                self._state["presets"] = {}
            self._state["presets"][name] = preset_data
            self._save()

    def delete_preset(self, name: str) -> bool:
        with self._lock:
            if name in self._state.get("presets", {}):
                del self._state["presets"][name]
                self._save()
                return True
            return False

    # ---------------------------------------------------------------
    # Model Health Cache (for recheck results)
    # ---------------------------------------------------------------
    def update_health_cache(self, health_data: Dict[str, Any]) -> None:
        with self._lock:
            self._state["model_health_cache"] = health_data
            self._state["last_recheck"] = time.time()
            self._save()

    def get_health_cache(self) -> Tuple[Dict[str, Any], float]:
        with self._lock:
            return (
                dict(self._state.get("model_health_cache", {})),
                self._state.get("last_recheck", 0),
            )

    # ---------------------------------------------------------------
    # Antigravity Auth Token / API Key
    # ---------------------------------------------------------------
    def get_antigravity_token(self) -> Optional[str]:
        with self._lock:
            return self._state.get("antigravity_token")

    def set_antigravity_token(self, token: Optional[str]) -> None:
        with self._lock:
            if token and token.strip():
                self._state["antigravity_token"] = token.strip()
            elif "antigravity_token" in self._state:
                del self._state["antigravity_token"]
            self._save()

    # ---------------------------------------------------------------
    # Full state access
    # ---------------------------------------------------------------
    def get_full_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def reload(self) -> None:
        with self._lock:
            self._state = self._load()


# Global singleton
state_manager = StateManager()
