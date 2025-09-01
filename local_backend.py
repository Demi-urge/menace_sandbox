from __future__ import annotations

"""Local LLM backend implementations for Ollama and vLLM servers.

The backends speak a very small REST dialect expected by the corresponding
servers.  They implement the :class:`LLMBackend` protocol so they can be used
with :class:`llm_interface.LLMClient`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, AsyncGenerator
import json
import os
import time

import requests
import rate_limit
import llm_config

try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore
except Exception:  # pragma: no cover - httpx may be provided as a stub in tests
    httpx = None  # type: ignore

from llm_interface import Prompt, Completion, LLMBackend, LLMClient


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

        cfg = llm_config.get_config()
        retries = cfg.max_retries
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)

        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            self._rate_limiter.consume(prompt_tokens)
            try:
                start = time.perf_counter()
                raw = self._post(payload)
                latency_ms = (time.perf_counter() - start) * 1000
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
            else:
                raw["backend"] = getattr(
                    self,
                    "backend_name",
                    self.__class__.__name__.replace("Backend", "").lower(),
                )
                raw.setdefault("model", self.model)
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
                    latency_ms=latency_ms,
                )

            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Failed to obtain completion from local backend")

    async def async_generate(
        self, prompt: Prompt
    ) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Asynchronously yield streamed chunks for *prompt*.

        The local REST servers used by the backends support server streaming
        where each line is a JSON object with a ``response`` field.  Some
        implementations, such as vLLM, use Server Sent Events and prefix each
        line with ``data:`` while others stream plain JSON.  This coroutine
        normalises both formats and yields the text portions as they arrive.
        """

        if httpx is None:  # pragma: no cover - dependency missing
            raise RuntimeError("httpx is required for async streaming")

        payload = {"model": self.model, "prompt": prompt.text, "stream": True}
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"

        cfg = llm_config.get_config()
        retries = cfg.max_retries
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)

        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            self._rate_limiter.consume(prompt_tokens)
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST", url, json=payload, timeout=30
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                data = line[len("data:"):].strip()
                                if data == "[DONE]":
                                    return
                            else:
                                data = line
                            try:
                                chunk = json.loads(data)
                            except Exception:
                                continue
                            if chunk.get("done"):
                                return
                            text = (
                                chunk.get("response")
                                or chunk.get("text")
                                or chunk.get("generated_text")
                                or chunk.get("delta", {}).get("content", "")
                            )
                            if not text:
                                continue
                            tokens = rate_limit.estimate_tokens(text, model=self.model)
                            self._rate_limiter.consume(tokens)
                            yield text
                        return
            except Exception:  # pragma: no cover - request failure
                if attempt == retries - 1:
                    raise
            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Failed to obtain streamed completion from local backend")


class OllamaBackend(_RESTBackend):
    """Backend speaking to an ``ollama`` model server."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        model = model or os.getenv("OLLAMA_MODEL", "mistral")
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.backend_name = "ollama"
        super().__init__(model=model, base_url=base_url, endpoint="api/generate")


class VLLMBackend(_RESTBackend):
    """Backend for a vLLM REST server."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        model = model or os.getenv("VLLM_MODEL", "llama3")
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.backend_name = "vllm"
        super().__init__(model=model, base_url=base_url, endpoint="generate")


def mixtral_client(model: str | None = None, base_url: str | None = None) -> LLMClient:
    """Return an :class:`LLMClient` using an :class:`OllamaBackend`.

    The *model* and *base_url* parameters fall back to ``OLLAMA_MODEL`` and
    ``OLLAMA_BASE_URL`` environment variables respectively.  When not provided
    the model defaults to ``"mixtral"``.
    """

    backend = OllamaBackend(
        model=model or os.getenv("OLLAMA_MODEL", "mixtral"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return LLMClient(model=backend.model, backends=[backend], log_prompts=False)


def llama3_client(model: str | None = None, base_url: str | None = None) -> LLMClient:
    """Return an :class:`LLMClient` using a :class:`VLLMBackend` configured for
    Llama 3."""

    backend = VLLMBackend(
        model=model or os.getenv("VLLM_MODEL", "llama3"),
        base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
    )
    return LLMClient(model=backend.model, backends=[backend], log_prompts=False)


__all__ = ["OllamaBackend", "VLLMBackend", "mixtral_client", "llama3_client"]
