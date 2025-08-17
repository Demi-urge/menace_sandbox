from __future__ import annotations

"""Light‑weight wrapper around the legacy :mod:`context_builder` module."""

import json
import time
from typing import Any, Dict, List

from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError


try:  # pragma: no cover - the legacy builder lives at repository root
    from context_builder import ContextBuilder as _LegacyContextBuilder  # type: ignore
except Exception:  # pragma: no cover - fallback when not available
    _LegacyContextBuilder = None  # type: ignore


class ContextBuilder:
    """Expose a ``build`` method compatible with older call sites."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _LegacyContextBuilder is None:  # pragma: no cover - defensive
            raise RuntimeError("context_builder module unavailable")
        self._builder = _LegacyContextBuilder(*args, **kwargs)
        self._cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    @log_and_measure
    def build(self, task_description: str, **kwargs: Any) -> str:
        """Return a compact JSON context for ``task_description``."""

        if not isinstance(task_description, str) or not task_description.strip():
            raise MalformedPromptError("task_description must be a non-empty string")

        assert self._builder is not None  # for type checkers
        retriever = getattr(self._builder, "retriever", None)
        if retriever is None:  # pragma: no cover - defensive
            raise VectorServiceError("retriever unavailable")

        attempts = 3
        backoff = 1.0
        hits: List[Any] = []
        for attempt in range(attempts):
            try:
                hits, _, _ = retriever.retrieve(task_description, top_k=1)
                break
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("vector search failed") from exc

        if not hits or max(getattr(h, "score", 0.0) for h in hits) < 0.1:
            cached = self._cache.get(task_description)
            if cached is not None:
                return cached
            return json.dumps({"note": "insufficient context"})

        context = ""
        backoff = 1.0
        for attempt in range(attempts):
            try:
                context = self._builder.build_context(task_description, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("context build failed") from exc

        self._cache[task_description] = context
        return context


__all__ = ["ContextBuilder"]

