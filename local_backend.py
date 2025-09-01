from __future__ import annotations

"""Local LLM backend implementations for Ollama and vLLM servers.

The backends speak a very small REST dialect expected by the corresponding
servers.  They implement the :class:`LLMBackend` protocol so they can be used
with :class:`llm_interface.LLMClient`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict
import os

import requests
import rate_limit
import llm_config

try:  # pragma: no cover - package vs module import
    from .retry_utils import with_retry
except Exception:  # pragma: no cover - fallback when not a package
    from retry_utils import with_retry

from llm_interface import Prompt, Completion, LLMBackend


@dataclass
class _RESTBackend(LLMBackend):
    """Shared helper for simple JSON-over-HTTP model servers."""

    model: str
    base_url: str
    endpoint: str
    _rate_limiter: rate_limit.TokenBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - simple initialiser
        cfg = llm_config.get_config()
        self._rate_limiter = rate_limit.TokenBucket(cfg.tokens_per_minute)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: Prompt) -> Completion:
        payload = {"model": self.model, "prompt": prompt.text}

        def do_request() -> Dict[str, Any]:
            return self._post(payload)

        cfg = llm_config.get_config()
        self._rate_limiter.update_rate(cfg.tokens_per_minute)
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)
        self._rate_limiter.consume(prompt_tokens)

        raw = with_retry(do_request, exc=requests.RequestException)
        text = (
            raw.get("text")
            or raw.get("response", "")
            or raw.get("generated_text", "")
        )
        completion_tokens = rate_limit.estimate_tokens(text, model=self.model)
        self._rate_limiter.consume(completion_tokens)
        return Completion(
            raw=raw,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OllamaBackend(_RESTBackend):
    """Backend speaking to an ``ollama`` model server."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        model = model or os.getenv("OLLAMA_MODEL", "mistral")
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        super().__init__(model=model, base_url=base_url, endpoint="api/generate")


class VLLMBackend(_RESTBackend):
    """Backend for a vLLM REST server."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        model = model or os.getenv("VLLM_MODEL", "llama3")
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        super().__init__(model=model, base_url=base_url, endpoint="generate")


__all__ = ["OllamaBackend", "VLLMBackend"]
