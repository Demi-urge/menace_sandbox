"""Minimal language model client interface with optional backend chaining.

This module defines a small dataclass :class:`LLMResult`, a light weight
``LLMBackend`` protocol and the :class:`LLMClient` helper used throughout the
codebase.  ``LLMClient`` can either be subclassed to implement a concrete
backend via :meth:`_generate` or instantiated with a list of backends to act as
an orchestrator with fallback behaviour.

The goal of keeping the abstraction surface tiny is to make it easy to plug in
new model providers while still supporting logging and simple response parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol, Sequence

import json
import requests
import time

try:  # pragma: no cover - package vs module import
    from . import llm_config, rate_limit
except Exception:  # pragma: no cover - stand-alone usage
    import llm_config  # type: ignore
    import rate_limit  # type: ignore

# ``Prompt`` lives in a separate module so other packages can import it without
# pulling in the entire client implementation.
try:  # pragma: no cover - package vs module import
    from .prompt_types import Prompt
except Exception:  # pragma: no cover - stand-alone usage
    from prompt_types import Prompt


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """Result of an LLM generation call."""

    raw: Dict[str, Any] | None = None
    text: str = ""
    parsed: Any | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None


# Backwards compatibility -------------------------------------------------
# ``Completion`` previously named the ``LLMResult`` container.  Keep an alias so
# older imports continue to work without modification.
Completion = LLMResult


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Minimal protocol for lightweight local backends."""

    model: str

    def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - interface
        ...


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------


class LLMClient:
    """Small helper class orchestrating language model backends.

    Parameters
    ----------
    model:
        Name of the model used for logging purposes.  When *backends* are
        provided and *model* is ``None`` the first backend's model name is used.
    backends:
        Optional list of backend instances.  When provided the client acts as a
        router trying each backend in order and falling back to the next one if
        an exception is raised.  This allows chaining a remote primary backend
        with a secondary local fallback.
    log_prompts:
        When ``True`` interactions are recorded in ``PromptDB``.  Logging is
        best effort and silently ignored if the database layer is unavailable.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        backends: Sequence[LLMBackend] | None = None,
        log_prompts: bool = True,
    ) -> None:
        if model is None and backends:
            model = backends[0].model
        if model is None:
            raise TypeError("model is required if backends are not provided")
        self.model = model
        self.backends = list(backends or [])
        self._log_prompts = log_prompts
        if log_prompts:
            try:  # pragma: no cover - database may not be available
                from prompt_db import PromptDB

                self.db = PromptDB(model=model)
            except Exception:  # pragma: no cover - logging is best effort
                self.db = None
        else:
            self.db = None

    # ------------------------------------------------------------------
    def _log(self, prompt: Prompt, result: LLMResult) -> None:
        """Persist *prompt* and *result* if logging is enabled."""

        if not self._log_prompts or not getattr(self, "db", None):
            return
        try:  # pragma: no cover - logging is best effort
            self.db.log(prompt, result)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _generate(self, prompt: Prompt) -> LLMResult:
        """Subclasses must implement this method."""

        raise NotImplementedError

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Prompt,
        *,
        parse_fn: Callable[[str], Any] | None = None,
    ) -> LLMResult:
        """Generate a completion for *prompt*.

        If the client was initialised with a list of backends each backend is
        attempted in order.  When ``prompt.metadata['small_task']`` is truthy the
        first backend is skipped which allows callers to favour a local backend
        for inexpensive tasks.  If all backends fail the last exception is
        raised.

        For subclassed clients without explicit backends, :meth:`_generate` is
        called directly.
        """

        # If explicit backends are configured act as a router
        if self.backends:
            order = list(self.backends)
            meta = getattr(prompt, "metadata", {})
            if meta.get("small_task") and len(order) > 1:
                # Prefer the secondary backend for small tasks
                order = order[1:] + order[:1]
            last_exc: Exception | None = None
            for backend in order:
                try:
                    result = backend.generate(prompt)
                except Exception as exc:  # pragma: no cover - backend failure
                    last_exc = exc
                    continue
                if parse_fn is not None:
                    try:
                        result.parsed = parse_fn(result.text)
                    except Exception:  # pragma: no cover - parsing is best effort
                        pass
                self._log(prompt, result)
                return result
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("no backends configured")

        # No explicit backends: delegate to subclass implementation
        result = self._generate(prompt)
        if parse_fn is not None:
            try:
                result.parsed = parse_fn(result.text)
            except Exception:  # pragma: no cover - parsing is best effort
                pass
        self._log(prompt, result)
        return result


__all__ = [
    "Prompt",
    "LLMResult",
    "Completion",
    "LLMClient",
    "LLMBackend",
    "OpenAIProvider",
    "requests",
    "time",
    "rate_limit",
]


class OpenAIProvider(LLMClient):
    """Minimal OpenAI Chat Completions client with retry/backoff."""

    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        max_retries: int | None = None,
    ) -> None:
        cfg = llm_config.get_config()
        model = model or cfg.model
        super().__init__(model)
        self.api_key = api_key or cfg.api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.max_retries = max_retries or cfg.max_retries
        self._session = requests.Session()
        self._rate_limiter = rate_limit.TokenBucket(cfg.tokens_per_minute)

    # ------------------------------------------------------------------
    def _generate(self, prompt: Prompt) -> LLMResult:
        messages: List[Dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        for ex in getattr(prompt, "examples", []):
            messages.append({"role": "system", "content": ex})
        messages.append({"role": "user", "content": prompt.user})

        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if getattr(prompt, "tags", None):
            payload["tags"] = prompt.tags
        if getattr(prompt, "vector_confidence", None) is not None:
            payload["vector_confidence"] = prompt.vector_confidence

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        cfg = llm_config.get_config()
        retries = cfg.max_retries
        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            tokens = rate_limit.estimate_tokens(
                " ".join(m.get("content", "") for m in payload["messages"]),
                model=self.model,
            )
            self._rate_limiter.consume(tokens)
            try:
                start = time.perf_counter()
                response = self._session.post(
                    self.api_url, headers=headers, json=payload, timeout=30
                )
                latency_ms = (time.perf_counter() - start) * 1000
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
            else:
                if response.status_code == 429 and attempt < retries - 1:
                    rate_limit.sleep_with_backoff(attempt)
                    continue
                if response.status_code >= 500 and attempt < retries - 1:
                    rate_limit.sleep_with_backoff(attempt)
                    continue
                response.raise_for_status()
                raw = response.json()
                text = ""
                try:
                    text = raw["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    pass
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    pass
                usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
                return LLMResult(
                    raw=raw,
                    text=text,
                    parsed=parsed,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    latency_ms=latency_ms,
                )

            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Failed to obtain completion from OpenAI")
