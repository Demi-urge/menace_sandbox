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
from typing import Any, Callable, Dict, List, Protocol, Sequence, AsyncGenerator

import asyncio
import json
import requests
import time

try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore
except Exception:  # pragma: no cover - httpx may be provided as a stub in tests
    httpx = None  # type: ignore

try:  # pragma: no cover - package vs module import
    from . import llm_config, rate_limit, llm_pricing
except Exception:  # pragma: no cover - stand-alone usage
    import llm_config  # type: ignore
    import rate_limit  # type: ignore
    import llm_pricing  # type: ignore

# ``Prompt`` lives in a separate module so other packages can import it without
# pulling in the entire client implementation.
try:  # pragma: no cover - package vs module import
    from .prompt_types import Prompt
except Exception:  # pragma: no cover - stand-alone usage
    from prompt_types import Prompt

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - allow stub during tests
    class ContextBuilder:  # type: ignore
        """Fallback stub for typing when context builder is unavailable."""
        pass


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
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
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

    def generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:  # pragma: no cover - interface
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
        # All clients share a single token bucket to coordinate rate limiting.
        cfg = llm_config.get_config()
        self._rate_limiter = rate_limit.SHARED_TOKEN_BUCKET
        self._rate_limiter.update_rate(getattr(cfg, "tokens_per_minute", 0))
        if log_prompts:
            try:  # pragma: no cover - database may not be available
                from prompt_db import PromptDB

                self.db = PromptDB(model=model)
            except Exception:  # pragma: no cover - logging is best effort
                self.db = None
        else:
            self.db = None

    # ------------------------------------------------------------------
    def _log(
        self, prompt: Prompt, result: LLMResult, *, backend: str | None = None
    ) -> None:
        """Persist *prompt* and *result* if logging is enabled."""

        if not self._log_prompts or not getattr(self, "db", None):
            return
        try:  # pragma: no cover - logging is best effort
            self.db.log(prompt, result, backend=backend)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _track_usage(self, builder: ContextBuilder, result: LLMResult) -> None:
        """Record token usage and retrieval metadata via the builder's tracker."""

        tracker = getattr(builder, "roi_tracker", None)
        if tracker is None:
            return
        usage = (result.raw or {}).get("usage") or {}
        metrics: Dict[str, Any] = {}
        if result.prompt_tokens is not None:
            metrics.setdefault("input_tokens", result.prompt_tokens)
        if result.completion_tokens is not None:
            metrics.setdefault("output_tokens", result.completion_tokens)
        if isinstance(usage, dict):
            metrics.setdefault("input_tokens", usage.get("input_tokens"))
            metrics.setdefault("output_tokens", usage.get("output_tokens"))
        retrieval_metrics = (result.raw or {}).get("retrieval_metrics")
        try:  # pragma: no cover - tracker is optional
            tracker.update(
                0.0,
                0.0,
                metrics=metrics or None,
                retrieval_metrics=retrieval_metrics,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:
        """Subclasses must implement this method."""

        raise NotImplementedError

    # ------------------------------------------------------------------
    async def _async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> AsyncGenerator[str, None]:
        """Asynchronous variant for streaming backends.

        Subclasses providing streaming capabilities should override this
        coroutine and ``yield`` chunks of the response as they arrive.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Prompt,
        *,
        parse_fn: Callable[[str], Any] | None = None,
        backend: str | None = None,
        context_builder: ContextBuilder,
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
            for backend_obj in order:
                try:
                    result = backend_obj.generate(
                        prompt, context_builder=context_builder
                    )
                except Exception as exc:  # pragma: no cover - backend failure
                    last_exc = exc
                    continue
                if parse_fn is not None:
                    try:
                        result.parsed = parse_fn(result.text)
                    except Exception:  # pragma: no cover - parsing is best effort
                        pass
                self._log(prompt, result, backend=getattr(backend_obj, "model", None))
                return result
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("no backends configured")

        # No explicit backends: delegate to subclass implementation
        result = self._generate(prompt, context_builder=context_builder)
        if parse_fn is not None:
            try:
                result.parsed = parse_fn(result.text)
            except Exception:  # pragma: no cover - parsing is best effort
                pass
        backend_name = backend or (result.raw or {}).get("backend")
        self._log(prompt, result, backend=backend_name)
        try:  # pragma: no cover - ROI tracking is best effort
            self._track_usage(context_builder, result)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    async def async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> AsyncGenerator[str, None]:
        """Asynchronously yield completion chunks for *prompt* and log the result."""

        cfg = llm_config.get_config()

        def _prompt_text(p: Prompt) -> str:
            parts = []
            if getattr(p, "system", None):
                parts.append(p.system)
            parts.extend(getattr(p, "examples", []))
            parts.append(getattr(p, "user", getattr(p, "text", "")))
            return " ".join(parts)

        def _finalize(
            text: str,
            *,
            backend_name: str | None,
            model: str,
            builder: ContextBuilder,
        ) -> LLMResult:
            prompt_tokens = rate_limit.estimate_tokens(_prompt_text(prompt), model=model)
            completion_tokens = rate_limit.estimate_tokens(text, model=model)
            in_rate, out_rate = llm_pricing.get_rates(model, cfg.pricing)
            cost = prompt_tokens * in_rate + completion_tokens * out_rate
            result = LLMResult(
                raw={
                    "backend": backend_name,
                    "model": model,
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "cost": cost,
                    },
                },
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=cost,
            )
            self._log(prompt, result, backend=backend_name)
            try:  # pragma: no cover - tracker optional
                self._track_usage(builder, result)
            except Exception:
                pass
            return result

        async def _stream(
            agen: AsyncGenerator[str, None],
            *,
            backend_name: str | None,
            model: str,
            builder: ContextBuilder,
        ) -> None:
            parts: List[str] = []
            while True:
                try:
                    chunk = await agen.__anext__()
                except StopAsyncIteration:
                    break
                parts.append(chunk)
                yield chunk
            text = "".join(parts)
            _finalize(text, backend_name=backend_name, model=model, builder=builder)

        if self.backends:
            order = list(self.backends)
            meta = getattr(prompt, "metadata", {})
            if meta.get("small_task") and len(order) > 1:
                order = order[1:] + order[:1]
            last_exc: Exception | None = None
            for backend in order:
                try:
                    agen_fn = backend.async_generate  # type: ignore[attr-defined]
                except AttributeError:  # pragma: no cover - backend lacks async
                    last_exc = RuntimeError("backend lacks async_generate")
                    continue
                try:
                    backend_name = getattr(backend, "backend_name", getattr(backend, "model", None))
                    model_name = getattr(backend, "model", self.model)
                    agen = agen_fn(prompt, context_builder=context_builder)
                    wrapper = _stream(
                        agen,
                        backend_name=backend_name,
                        model=model_name,
                        builder=context_builder,
                    )
                    async for chunk in wrapper:
                        yield chunk
                    return
                except Exception as exc:  # pragma: no cover - backend failure
                    last_exc = exc
                    continue
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("no backends configured")

        backend_name = getattr(self, "backend_name", None)
        agen = self._async_generate(prompt, context_builder=context_builder)
        wrapper = _stream(
            agen, backend_name=backend_name, model=self.model, builder=context_builder
        )
        async for chunk in wrapper:
            yield chunk
        return


__all__ = [
    "Prompt",
    "LLMResult",
    "Completion",
    "LLMClient",
    "LLMBackend",
    "OpenAIProvider",
    "requests",
    "httpx",
    "asyncio",
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
        self.backend_name = "openai"

    # ------------------------------------------------------------------
    def _prepare_payload(self, prompt: Prompt) -> Dict[str, Any]:
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
        return payload

    # ------------------------------------------------------------------
    def _generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:
        payload = self._prepare_payload(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        cfg = llm_config.get_config()
        retries = cfg.max_retries
        prompt_tokens_est = rate_limit.estimate_tokens(
            " ".join(m.get("content", "") for m in payload["messages"]),
            model=self.model,
        )
        for attempt in range(retries):
            self._rate_limiter.update_rate(cfg.tokens_per_minute)
            self._rate_limiter.consume(prompt_tokens_est)
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
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens_est
                completion_tokens = usage.get("completion_tokens") or rate_limit.estimate_tokens(
                    text, model=self.model
                )
                input_tokens = usage.get("input_tokens") or prompt_tokens
                output_tokens = usage.get("output_tokens") or completion_tokens
                in_rate, out_rate = llm_pricing.get_rates(self.model, cfg.pricing)
                cost = input_tokens * in_rate + output_tokens * out_rate
                total = (prompt_tokens or 0) + (completion_tokens or 0)
                extra = max(0, total - prompt_tokens_est)
                if extra:
                    self._rate_limiter.consume(extra)
                usage.setdefault("input_tokens", input_tokens)
                usage.setdefault("output_tokens", output_tokens)
                usage["cost"] = cost
                raw["usage"] = usage
                raw.setdefault("model", self.model)
                raw["backend"] = "openai"
                return LLMResult(
                    raw=raw,
                    text=text,
                    parsed=parsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                )

            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Failed to obtain completion from OpenAI")

    # ------------------------------------------------------------------
    async def _async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> AsyncGenerator[str, None]:
        if httpx is None:  # pragma: no cover - dependency missing
            raise RuntimeError("httpx is required for async streaming")

        payload = self._prepare_payload(prompt)
        payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        cfg = llm_config.get_config()
        retries = cfg.max_retries
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST", self.api_url, headers=headers, json=payload, timeout=30
                    ) as response:
                        if response.status_code == 429 and attempt < retries - 1:
                            rate_limit.sleep_with_backoff(attempt)
                            continue
                        if response.status_code >= 500 and attempt < retries - 1:
                            rate_limit.sleep_with_backoff(attempt)
                            continue
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            if not line.startswith("data:"):
                                continue
                            data = line[len("data:"):].strip()
                            if data == "[DONE]":
                                return
                            try:
                                chunk = json.loads(data)
                            except Exception:
                                continue
                            try:
                                delta = chunk["choices"][0]["delta"].get("content", "")
                            except Exception:
                                delta = ""
                            if delta:
                                yield delta
                        return
            except Exception:  # pragma: no cover - request failure
                if attempt == retries - 1:
                    raise

            rate_limit.sleep_with_backoff(attempt)

        raise RuntimeError("Failed to obtain completion from OpenAI")

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Prompt,
        *,
        parse_fn: Callable[[str], Any] | None = None,
    ) -> LLMResult | asyncio.Task[LLMResult]:
        """Synchronously generate a completion aggregating streamed chunks.

        When called while an event loop is already running the collection is
        scheduled on that loop and a task is returned which must be awaited by
        the caller.  If no loop is active ``asyncio.run`` is used and the result
        is returned directly.
        """

        payload = self._prepare_payload(prompt)
        prompt_tokens = rate_limit.estimate_tokens(
            " ".join(m.get("content", "") for m in payload["messages"]),
            model=self.model,
        )
        cfg = llm_config.get_config()
        self._rate_limiter.update_rate(cfg.tokens_per_minute)
        self._rate_limiter.consume(prompt_tokens)

        async def collect() -> LLMResult:
            text_parts: List[str] = []
            start = time.perf_counter()
            async for part in self._async_generate(prompt):
                text_parts.append(part)
            latency_ms = (time.perf_counter() - start) * 1000
            text = "".join(text_parts)
            completion_tokens = rate_limit.estimate_tokens(text, model=self.model)
            self._rate_limiter.consume(completion_tokens)
            in_rate, out_rate = llm_pricing.get_rates(self.model, cfg.pricing)
            cost = prompt_tokens * in_rate + completion_tokens * out_rate
            parsed = None
            if parse_fn is not None:
                try:
                    parsed = parse_fn(text)
                except Exception:  # pragma: no cover - best effort
                    parsed = None

            result = LLMResult(
                raw={
                    "backend": "openai",
                    "model": self.model,
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "cost": cost,
                    },
                },
                text=text,
                parsed=parsed,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost=cost,
            )
            self._log(prompt, result, backend="openai")
            return result

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(collect())
        else:
            return loop.create_task(collect())


# Backwards compatibility -------------------------------------------------
# ``OpenAILLMClient`` previously referenced the OpenAI provider implementation.
# Keep an alias so older imports continue to function after consolidation.
OpenAILLMClient = OpenAIProvider
__all__.append("OpenAILLMClient")
