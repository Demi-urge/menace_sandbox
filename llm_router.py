from __future__ import annotations

"""Routing helpers for language model backends."""

from typing import Dict, Tuple, AsyncGenerator
import time

from llm_interface import Prompt, LLMResult, LLMClient
from llm_registry import create_backend, register_backend_from_path
from sandbox_settings import SandboxSettings
from rate_limit import estimate_tokens
from prompt_db import PromptDB

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - allow stub in tests
    class ContextBuilder:  # type: ignore
        pass


class LLMRouter(LLMClient):
    """Route requests between remote and local backends.

    The router chooses the remote backend for larger prompts while favouring
    the local backend for smaller ones.  Metadata such as ROI tags can override
    this behaviour.  Backends with recent failures are avoided.  The chosen
    backend is logged to ``PromptDB`` for post-hoc analysis.
    """

    def __init__(
        self,
        remote: LLMClient,
        local: LLMClient,
        *,
        size_threshold: int = 1000,
        failure_cooldown: float = 60.0,
    ) -> None:
        super().__init__("router", log_prompts=False)
        self.remote = remote
        self.local = local
        self.size_threshold = size_threshold
        self.failure_cooldown = failure_cooldown
        self._last_failure: Dict[LLMClient, float] = {}
        # prevent duplicate logging by underlying clients
        for backend in (remote, local):
            if isinstance(backend, LLMClient):
                backend._log_prompts = False  # type: ignore[attr-defined]
        try:  # pragma: no cover - logging is best effort
            self.db = PromptDB(model="router")
        except Exception:  # pragma: no cover - optional
            self.db = None

    def _recent_failure(self, backend: LLMClient) -> bool:
        ts = self._last_failure.get(backend)
        return ts is not None and (time.time() - ts) < self.failure_cooldown

    def _select_backends(self, prompt: Prompt) -> Tuple[LLMClient, LLMClient]:
        tokens = estimate_tokens(prompt.user)
        tags = set(getattr(prompt, "tags", []))
        meta = getattr(prompt, "metadata", {})
        tags.update(meta.get("tags", []))

        # Base selection on ROI tags and prompt size
        if "low_roi" in tags:
            primary = self.local
        elif "high_roi" in tags:
            primary = self.remote
        else:
            primary = self.remote if tokens > self.size_threshold else self.local

        # Adjust based on vector confidence – high confidence prefers remote
        vc = getattr(prompt, "vector_confidence", None)
        if vc is not None:
            if vc >= 0.8:
                primary = self.remote
            elif vc <= 0.2:
                primary = self.local

        # Incorporate estimated token cost
        try:
            from llm_pricing import get_input_rate

            cost_remote = tokens * get_input_rate(self.remote.model)
            cost_local = tokens * get_input_rate(self.local.model)
            if primary is self.remote and cost_remote > cost_local:
                primary = self.local
            elif primary is self.local and cost_local > cost_remote:
                primary = self.remote
        except Exception:  # pragma: no cover - pricing is best effort
            pass

        # Consider recent latency from PromptDB if available
        def _avg_latency(backend: LLMClient) -> float | None:
            db = getattr(self, "db", None)
            if not db or not hasattr(db, "fetch_logs"):
                return None
            try:
                entries = db.fetch_logs(limit=5, model=backend.model)
            except Exception:  # pragma: no cover - best effort
                return None
            lats = [e.get("latency_ms") for e in entries if e.get("latency_ms") is not None]
            if not lats:
                return None
            return sum(lats) / len(lats)

        lat_primary = _avg_latency(primary)
        other = self.local if primary is self.remote else self.remote
        lat_other = _avg_latency(other)
        if lat_primary is not None and lat_other is not None and lat_primary > lat_other:
            primary = other

        # Avoid recently failing backend
        if self._recent_failure(primary):
            primary = self.local if primary is self.remote else self.remote

        fallback = self.local if primary is self.remote else self.remote
        return primary, fallback

    def _generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> LLMResult:
        primary, fallback = self._select_backends(prompt)
        chosen = primary
        try:
            result = primary.generate(prompt, context_builder=context_builder)
        except Exception:
            self._last_failure[primary] = time.time()
            result = fallback.generate(prompt, context_builder=context_builder)
            chosen = fallback
        if getattr(self, "db", None):  # pragma: no cover - logging is best effort
            result.raw = dict(result.raw or {})
            result.raw.setdefault("model", chosen.model)
            result.raw["backend"] = chosen.model
            if getattr(prompt, "tags", None):
                result.raw.setdefault("tags", list(prompt.tags))
            if getattr(prompt, "vector_confidence", None) is not None:
                result.raw.setdefault("vector_confidence", prompt.vector_confidence)
            try:
                self.db.log(prompt, result, backend=chosen.model)
            except Exception:
                pass
        return result

    async def async_generate(
        self, prompt: Prompt, *, context_builder: ContextBuilder
    ) -> AsyncGenerator[str, None]:
        """Asynchronously stream chunks from the chosen backend with fallback."""

        primary, fallback = self._select_backends(prompt)
        chosen = primary
        chunks: list[str] = []

        async def _run(backend: LLMClient):
            agen = backend.async_generate  # type: ignore[attr-defined]
            async for part in agen(prompt, context_builder=context_builder):
                chunks.append(part)
                yield part

        try:
            async for part in _run(primary):
                yield part
        except Exception:
            self._last_failure[primary] = time.time()
            chunks.clear()
            chosen = fallback
            async for part in _run(fallback):
                yield part

        if getattr(self, "db", None):  # pragma: no cover - logging is best effort
            raw = {"backend": chosen.model, "model": chosen.model}
            if getattr(prompt, "tags", None):
                raw["tags"] = list(prompt.tags)
            if getattr(prompt, "vector_confidence", None) is not None:
                raw["vector_confidence"] = prompt.vector_confidence
            result = LLMResult(raw=raw, text="".join(chunks))
            try:
                self.db.log(prompt, result, backend=chosen.model)
            except Exception:
                pass


def client_from_settings(
    settings: SandboxSettings | None = None, *, size_threshold: int = 1000
) -> LLMClient:
    """Create an :class:`LLMClient` based on :class:`SandboxSettings`.

    ``SandboxSettings.available_backends`` maps backend names to dotted import
    paths referencing factories or classes returning :class:`LLMClient`
    instances.  Entries in this mapping are automatically registered with
    :mod:`llm_registry` allowing the router to instantiate them by name.  The
    ``preferred_llm_backend`` selects the primary backend while
    ``llm_fallback_backend`` optionally specifies a secondary backend used by
    :class:`LLMRouter` as a fallback.
    """
    settings = settings or SandboxSettings()

    # Populate registry from settings so custom backends can be provided via
    # configuration without touching the codebase.
    for name, path in settings.available_backends.items():
        register_backend_from_path(name, path)

    backend = (settings.preferred_llm_backend or settings.llm_backend).lower()
    fallback = settings.llm_fallback_backend
    fallback = fallback.lower() if fallback else None

    primary = create_backend(backend)
    if not fallback:
        return primary
    fallback_client = create_backend(fallback)
    return LLMRouter(remote=primary, local=fallback_client, size_threshold=size_threshold)


__all__ = ["LLMRouter", "client_from_settings"]
