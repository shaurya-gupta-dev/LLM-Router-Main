"""Base provider classes and unified response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        provider: str = "",
        is_429: bool = False,
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.is_429 = is_429
        self.diagnostics = diagnostics or {}


class RateLimitException(ProviderException):
    def __init__(self, message: str, provider: str = "", retry_after: float = 60.0):
        super().__init__(message=message, status_code=429, provider=provider, is_429=True)
        self.retry_after = retry_after


class AuthException(ProviderException):
    def __init__(self, message: str, provider: str = ""):
        super().__init__(message=message, status_code=401, provider=provider)


class UnifiedChatResult(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    def __init__(self, provider_id: str, base_url: str):
        self.provider_id = provider_id
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        api_key: str,
    ) -> UnifiedChatResult:
        """Execute a non-streaming chat completion request."""
        pass

    @abstractmethod
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        temperature: float,
        api_key: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a streaming chat completion request.
        Yields unified chunks: {"text": str, "tool_call": dict, "stop_reason": str, "output_tokens": int}
        """
        pass
