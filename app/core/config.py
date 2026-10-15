"""Configuration loader and schema models for LLM Router."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ProviderModelConfig(BaseModel):
    id: str
    name: str
    context_window: int = 128000
    supports_tools: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False


class FreeTierConfig(BaseModel):
    rpm: int = 15
    rpd: int = 1000
    tpm: int = 100000


class ProviderDefinition(BaseModel):
    name: str
    type: str  # "gemini" | "openai_compat"
    base_url: str
    env_keys: List[str] = Field(default_factory=list)
    docs_url: str = ""
    free_tier: FreeTierConfig = Field(default_factory=FreeTierConfig)
    models: List[ProviderModelConfig] = Field(default_factory=list)


class FallbackStep(BaseModel):
    provider: str
    model: str


class ModelAliasConfig(BaseModel):
    display_name: str
    fallbacks: List[FallbackStep] = Field(default_factory=list)


class RoutingSettings(BaseModel):
    default_timeout_seconds: int = 60
    max_fallback_attempts: int = 8
    cooldown_seconds_on_429: int = 60
    cooldown_seconds_on_5xx: int = 30
    retry_backoff_factor: float = 1.5


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Master Key for incoming requests
    master_key: str = Field(default="llm-router-secret-key-12345", alias="MASTER_KEY")
    router_host: str = Field(default="0.0.0.0", alias="ROUTER_HOST")
    router_port: int = Field(default=8080, alias="ROUTER_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # 14+ Provider API keys (can be comma-separated for multi-key pooling!)
    gemini_api_keys: str = Field(default="", alias="GEMINI_API_KEYS")
    groq_api_keys: str = Field(default="", alias="GROQ_API_KEYS")
    github_tokens: str = Field(default="", alias="GITHUB_TOKENS")
    cerebras_api_keys: str = Field(default="", alias="CEREBRAS_API_KEYS")
    openrouter_api_keys: str = Field(default="", alias="OPENROUTER_API_KEYS")
    nvidia_api_keys: str = Field(default="", alias="NVIDIA_API_KEYS")
    huggingface_api_keys: str = Field(default="", alias="HUGGINGFACE_API_KEYS")
    cloudflare_api_keys: str = Field(default="", alias="CLOUDFLARE_API_KEYS")
    cohere_api_keys: str = Field(default="", alias="COHERE_API_KEYS")
    sambanova_api_keys: str = Field(default="", alias="SAMBANOVA_API_KEYS")
    together_api_keys: str = Field(default="", alias="TOGETHER_API_KEYS")
    kilo_api_keys: str = Field(default="", alias="KILO_API_KEYS")
    llm7_api_keys: str = Field(default="", alias="LLM7_API_KEYS")
    deepseek_api_keys: str = Field(default="", alias="DEEPSEEK_API_KEYS")
    mistral_api_keys: str = Field(default="", alias="MISTRAL_API_KEYS")
    zai_api_keys: str = Field(default="", alias="ZAI_API_KEYS")



from dotenv import dotenv_values, load_dotenv


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RouterConfig:
    def __init__(self):
        self.settings = AppSettings()
        self.providers: Dict[str, ProviderDefinition] = {}
        self.models: Dict[str, ModelAliasConfig] = {}
        self.routing = RoutingSettings()
        self.env_cache: Dict[str, str] = {}
        self.load()

    def load(self):
        # 0. Dynamically read .env freshly from disk
        env_path = BASE_DIR / ".env"
        load_dotenv(env_path, override=True)
        raw_env = dotenv_values(env_path)
        self.env_cache = {
            k.upper(): v.strip() for k, v in raw_env.items() if v and v.strip()
        }
        for k, v in self.env_cache.items():
            os.environ[k] = v

        # 1. Load providers.yaml
        prov_data = load_yaml(BASE_DIR / "config" / "providers.yaml")
        for key, raw in prov_data.get("providers", {}).items():
            self.providers[key] = ProviderDefinition(**raw)

        # 2. Load config.yaml
        cfg_data = load_yaml(BASE_DIR / "config" / "config.yaml")
        if "routing" in cfg_data:
            self.routing = RoutingSettings(**cfg_data["routing"])

        for alias, raw in cfg_data.get("models", {}).items():
            self.models[alias] = ModelAliasConfig(**raw)

    def get_provider_keys(self, provider_id: str) -> List[str]:
        """Extract all configured keys for a provider from env variables."""
        prov = self.providers.get(provider_id)
        if not prov:
            return []

        keys: List[str] = []
        for env_var in prov.env_keys:
            ev_upper = env_var.upper()
            val = (
                self.env_cache.get(ev_upper, "")
                or os.environ.get(env_var, "")
                or os.environ.get(ev_upper, "")
            ).strip()

            if not val:
                attr = env_var.lower()
                val = getattr(self.settings, attr, "") if hasattr(self.settings, attr) else ""
                val = str(val).strip()

            if val:
                for k in val.split(","):
                    clean = k.strip().strip("'\"")
                    if clean and clean not in keys:
                        keys.append(clean)
        return keys


# Global singleton instance
config = RouterConfig()
