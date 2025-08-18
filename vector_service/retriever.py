from __future__ import annotations

"""Wrapper around :class:`universal_retriever.UniversalRetriever`.

The real retriever object lives in :mod:`universal_retriever` and returns
``ResultBundle`` instances.  The :class:`Retriever` below normalises those
results into plain dictionaries and provides a tiny layer of retry and
heuristic fallbacks used across the code base.
"""

from dataclasses import dataclass, field
import time
import asyncio
from typing import Any, Dict, Iterable, List, Sequence

from redaction_utils import redact_dict as pii_redact_dict, redact_text
from security.secret_redactor import redact, redact_dict as secret_redact_dict
from license_detector import detect as license_detect
from analysis.semantic_diff_filter import find_semantic_risks
from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError


try:  # pragma: no cover - optional dependency
    from universal_retriever import UniversalRetriever, ResultBundle  # type: ignore
except Exception:  # pragma: no cover - fallback when not available
    UniversalRetriever = None  # type: ignore
    ResultBundle = object  # type: ignore


@dataclass
class FallbackResult(Sequence[Dict[str, Any]]):
    """Container returned when search falls back to heuristics.

    The object behaves like a sequence of result dictionaries while also
    exposing the ``reason`` and the confidence of the failed lookup.
    """

    reason: str
    results: List[Dict[str, Any]]
    confidence: float = 0.0

    def __iter__(self):  # pragma: no cover - simple delegation
        return iter(self.results)

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.results)

    def __getitem__(self, item: int) -> Dict[str, Any]:  # pragma: no cover
        return self.results[item]


@dataclass
class Retriever:
    """Facade around :class:`UniversalRetriever` with caching and retries.

    When the underlying retriever exposes ``retrieve_with_confidence`` the
    confidence score is used to determine whether to retry the search with
    broader parameters.  If all attempts fail a :class:`FallbackResult` is
    returned containing the reason and heuristic hits.
    """

    retriever: UniversalRetriever | None = None
    top_k: int = 5
    similarity_threshold: float = 0.1
    retriever_kwargs: Dict[str, Any] = field(default_factory=dict)
    content_filtering: bool = field(default=True)
    _cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _get_retriever(self) -> UniversalRetriever:
        if self.retriever is None:
            if UniversalRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("UniversalRetriever unavailable")
            self.retriever = UniversalRetriever(**self.retriever_kwargs)
        return self.retriever

    # ------------------------------------------------------------------
    def _parse_hits(self, hits: Iterable[Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for h in hits:
            meta = getattr(h, "metadata", {})
            if not isinstance(meta, dict) or not meta.get("redacted"):
                continue
            if isinstance(h, ResultBundle):
                item = h.to_dict()
                item["record_id"] = getattr(h, "record_id", None)
                meta = item.get("metadata", meta)
            elif hasattr(h, "to_dict"):
                item = h.to_dict()
                item.setdefault("record_id", getattr(h, "record_id", None))
                meta = item.get("metadata", meta)
            else:  # pragma: no cover - very defensive
                item = {
                    "origin_db": getattr(h, "origin_db", ""),
                    "record_id": getattr(h, "record_id", None),
                    "score": getattr(h, "score", 0.0),
                    "reason": getattr(h, "reason", ""),
                    "metadata": meta,
                }
            text = str(item.get("text") or "")
            lic = license_detect(text)
            if lic:
                continue
            alerts = find_semantic_risks(text.splitlines())
            if alerts:
                meta.setdefault("semantic_alerts", alerts)
            results.append(secret_redact_dict(pii_redact_dict(item)))
        return results

    # ------------------------------------------------------------------
    def _fallback(self, reason: str) -> List[Dict[str, Any]]:
        reason = redact(redact_text(reason))
        return [
            {
                "origin_db": "heuristic",
                "record_id": None,
                "score": 0.0,
                "metadata": {},
                "reason": reason,
            }
        ]

    # ------------------------------------------------------------------
    @log_and_measure
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        session_id: str = "",
        similarity_threshold: float | None = None,
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Perform semantic search and return normalised results.

        The method first attempts ``retrieve_with_confidence`` when available.
        If the returned confidence is below ``similarity_threshold`` or no
        results are produced, the search is retried once with ``top_k`` doubled.
        After failing again a :class:`FallbackResult` is returned instead of
        plain hits.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        query = redact(redact_text(query))
        k = top_k or self.top_k
        thresh = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        retriever = self._get_retriever()

        attempts = 2
        backoff = 1.0
        hits: List[Any] = []
        confidence = 0.0
        for attempt in range(attempts):
            try:
                if hasattr(retriever, "retrieve_with_confidence"):
                    hits, confidence, _ = retriever.retrieve_with_confidence(  # type: ignore[attr-defined]
                        query, top_k=k
                    )
                else:
                    hits, _, _ = retriever.retrieve(query, top_k=k)  # type: ignore[arg-type]
                    confidence = (
                        max(getattr(h, "score", 0.0) for h in hits) if hits else 0.0
                    )
            except AttributeError:  # pragma: no cover - compatibility fallback
                try:
                    hits = retriever.search(query)[:k]  # type: ignore[assignment]
                    confidence = (
                        max(getattr(h, "score", 0.0) for h in hits) if hits else 0.0
                    )
                except Exception as exc:
                    raise VectorServiceError("vector search failed") from exc
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("vector search failed") from exc

            if hits and confidence >= thresh:
                results = self._parse_hits(hits)
                if results:
                    self._cache[query] = results
                    return results

            # Broaden parameters and retry once
            k *= 2

        cached = self._cache.get(query)
        if cached is not None:
            return cached
        fb_hits = self._fallback("no results" if not hits else "low confidence")
        return FallbackResult(
            "no results" if not hits else "low confidence", fb_hits, confidence
        )

    # ------------------------------------------------------------------
    @log_and_measure
    async def search_async(
        self,
        query: str,
        *,
        top_k: int | None = None,
        session_id: str = "",
        similarity_threshold: float | None = None,
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Asynchronous wrapper for :meth:`search`.

        Executes the synchronous :meth:`search` implementation in a separate
        thread so it can be awaited without blocking the event loop.
        """
        query = redact(redact_text(query))
        return await asyncio.to_thread(
            self.search.__wrapped__,
            self,
            query,
            top_k=top_k,
            session_id=session_id,
            similarity_threshold=similarity_threshold,
        )

    # ------------------------------------------------------------------
    def error_frequency(self, error_id: int) -> float:
        """Expose raw error frequency metric from the underlying service."""

        try:
            return float(self._get_retriever()._error_frequency(int(error_id)))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def workflow_usage(self, wf: Any) -> float:
        """Expose workflow usage count for a workflow record."""

        try:
            return float(self._get_retriever()._workflow_usage(wf))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def bot_deploy_freq(self, bot_id: int) -> float:
        """Return deployment frequency for the given bot id."""

        try:
            return float(self._get_retriever()._bot_deploy_freq(int(bot_id)))
        except Exception:
            return 0.0


__all__ = ["Retriever", "FallbackResult"]

