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
from llm_interface import Prompt, Completion, LLMBackend, LLMClient
try:  # pragma: no cover - optional during tests
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - allow stub
    class ContextBuilder:  # type: ignore
        pass

try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore
except Exception:  # pragma: no cover - httpx may be provided as a stub in tests
    httpx = None  # type: ignore


class _RetryableHTTPError(requests.HTTPError):
    """Error raised for HTTP 5xx responses to trigger retry."""


@dataclass
class _RESTBackend(LLMBackend):
    """Shared helper for simple JSON-over-HTTP model servers."""

    model: str
    base_url: str
    endpoint: str
    _rate_limiter: rate_limit.TokenBucket = field(init=False, repr=False)
    _session: requests.Session = field(default_factory=requests.Session, init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - simple initialiser
        cfg = llm_config.get_config()
        self._rate_limiter = rate_limit.SHARED_TOKEN_BUCKET
        self._rate_limiter.update_rate(getattr(cfg, "tokens_per_minute", 0))

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        response = self._session.post(url, json=payload, timeout=30)
        if 500 <= response.status_code < 600:
            raise _RetryableHTTPError(f"server error: {response.status_code}", response=response)
        response.raise_for_status()
        return response.json()

    def generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> Completion:
        payload = {"model": self.model, "prompt": prompt.text}
        if getattr(prompt, "tags", None):
            payload["tags"] = list(prompt.tags)
        if getattr(prompt, "vector_confidence", None) is not None:
            payload["vector_confidence"] = prompt.vector_confidence

        cfg = llm_config.get_config()
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)

        self._rate_limiter.update_rate(cfg.tokens_per_minute)

        raw: Dict[str, Any] | None = None
        latency_ms = 0.0
        for attempt in range(cfg.max_retries):
            try:
                start = time.perf_counter()
                raw = self._post(payload)
                latency_ms = (time.perf_counter() - start) * 1000
                break
            except (requests.RequestException, _RetryableHTTPError):
                if attempt == cfg.max_retries - 1:
                    raise
                rate_limit.sleep_with_backoff(attempt)
        if raw is None:
            raise RuntimeError("local backend request failed")

        self._rate_limiter.consume(prompt_tokens)

        raw["backend"] = getattr(
            self,
            "backend_name",
            self.__class__.__name__.replace("Backend", "").lower(),
        )
        raw.setdefault("model", self.model)
        if getattr(prompt, "tags", None):
            raw.setdefault("tags", list(prompt.tags))
        if getattr(prompt, "vector_confidence", None) is not None:
            raw.setdefault("vector_confidence", prompt.vector_confidence)
        text = (
            raw.get("text")
            or raw.get("response", "")
            or raw.get("generated_text", "")
        )
        completion_tokens = rate_limit.estimate_tokens(text, model=self.model)
        self._rate_limiter.consume(completion_tokens)
        raw.setdefault(
            "usage",
            {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "cost": 0.0,
            },
        )
        return Completion(
            raw=raw,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=0.0,
            latency_ms=latency_ms,
        )

    async def async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
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
        if getattr(prompt, "tags", None):
            payload["tags"] = list(prompt.tags)
        if getattr(prompt, "vector_confidence", None) is not None:
            payload["vector_confidence"] = prompt.vector_confidence
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"

        cfg = llm_config.get_config()
        retries = cfg.max_retries
        prompt_tokens = rate_limit.estimate_tokens(prompt.text, model=self.model)

        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST", url, json=payload, timeout=30
                    ) as response:
                        response.raise_for_status()
                        self._rate_limiter.consume(prompt_tokens)
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

    async def async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from the Ollama server."""
        async for chunk in super().async_generate(
            prompt, context_builder=context_builder
        ):
            yield chunk


class VLLMBackend(_RESTBackend):
    """Backend for a vLLM REST server."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        model = model or os.getenv("VLLM_MODEL", "llama3")
        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.backend_name = "vllm"
        super().__init__(model=model, base_url=base_url, endpoint="generate")

    async def async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from the vLLM server."""
        async for chunk in super().async_generate(
            prompt, context_builder=context_builder
        ):
            yield chunk


def mixtral_client(
    model: str | None = None,
    base_url: str | None = None,
    *,
    log_prompts: bool = True,
) -> LLMClient:
    """Return an :class:`LLMClient` using an :class:`OllamaBackend`.

    The *model* and *base_url* parameters fall back to ``OLLAMA_MODEL`` and
    ``OLLAMA_BASE_URL`` environment variables respectively.  When not provided
    the model defaults to ``"mixtral"``.
    """

    backend = OllamaBackend(
        model=model or os.getenv("OLLAMA_MODEL", "mixtral"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return LLMClient(model=backend.model, backends=[backend], log_prompts=log_prompts)


def llama3_client(
    model: str | None = None,
    base_url: str | None = None,
    *,
    log_prompts: bool = True,
) -> LLMClient:
    """Return an :class:`LLMClient` using a :class:`VLLMBackend` configured for
    Llama 3."""

    backend = VLLMBackend(
        model=model or os.getenv("VLLM_MODEL", "llama3"),
        base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
    )
    return LLMClient(model=backend.model, backends=[backend], log_prompts=log_prompts)


__all__ = ["OllamaBackend", "VLLMBackend", "mixtral_client", "llama3_client"]
