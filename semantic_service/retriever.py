from __future__ import annotations

"""Wrapper around :class:`universal_retriever.UniversalRetriever`.

The real retriever object lives in :mod:`universal_retriever` and returns
``ResultBundle`` instances.  The :class:`Retriever` below normalises those
results into plain dictionaries and provides a tiny layer of retry and
heuristic fallbacks used across the code base.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterable, List, Sequence

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
    _cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _get_retriever(self) -> UniversalRetriever:
        if self.retriever is None:
            if UniversalRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("UniversalRetriever unavailable")
            self.retriever = UniversalRetriever()
        return self.retriever

    # ------------------------------------------------------------------
    def _parse_hits(self, hits: Iterable[Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for h in hits:
            if isinstance(h, ResultBundle):
                item = h.to_dict()
                item["record_id"] = getattr(h, "record_id", None)
            elif hasattr(h, "to_dict"):
                item = h.to_dict()
                item.setdefault("record_id", getattr(h, "record_id", None))
            else:  # pragma: no cover - very defensive
                meta = getattr(h, "metadata", {})
                item = {
                    "origin_db": getattr(h, "origin_db", ""),
                    "record_id": getattr(h, "record_id", None),
                    "score": getattr(h, "score", 0.0),
                    "reason": getattr(h, "reason", ""),
                    "metadata": meta,
                }
            results.append(item)
        return results

    # ------------------------------------------------------------------
    def _fallback(self, reason: str) -> List[Dict[str, Any]]:
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


__all__ = ["Retriever", "FallbackResult"]

