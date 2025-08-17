"""Cross-database context builder used by language model prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio

from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError
from .retriever import Retriever, FallbackResult
from config import ContextBuilderConfig

# Alias retained for backward compatibility with tests expecting
# ``UniversalRetriever`` to be injectable.
UniversalRetriever = Retriever

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from . import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when undefined
    class ErrorResult(Exception):
        """Fallback ErrorResult used when vector service lacks explicit class."""

        pass

# Optional summariser -------------------------------------------------------
try:  # pragma: no cover - heavy dependency
    from menace_memory_manager import MenaceMemoryManager, _summarise_text
except Exception:  # pragma: no cover - tiny fallback helper
    MenaceMemoryManager = None  # type: ignore

    def _summarise_text(text: str, ratio: float = 0.3) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= 120:
            return text
        return text[:117] + "..."


@dataclass
class _ScoredEntry:
    entry: Dict[str, Any]
    score: float


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
        db_weights: Dict[str, float] | None = None,
        max_tokens: int = ContextBuilderConfig().max_tokens,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.memory = memory_manager
        self._cache: Dict[Tuple[str, int], str] = {}
        self.db_weights = db_weights or {}
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    def _summarise(self, text: str) -> str:
        if self.memory and hasattr(self.memory, "_summarise_text"):
            try:
                return self.memory._summarise_text(text)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - fallback
                pass
        return _summarise_text(text)

    # ------------------------------------------------------------------
    def _metric(self, origin: str, meta: Dict[str, Any]) -> float | None:
        """Extract ROI/success metrics from metadata."""

        try:
            if origin == "error":
                freq = meta.get("frequency")
                if freq is not None:
                    return 1.0 / (1.0 + float(freq))

            if origin == "bot":
                for key in ("roi", "deploy_count"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "workflow":
                for key in ("roi", "usage", "runs"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "code":
                for key in ("roi", "patch_success"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "discrepancy":
                for key in ("roi", "severity", "impact"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    # ------------------------------------------------------------------
    def _bundle_to_entry(self, bundle: Dict[str, Any]) -> Tuple[str, _ScoredEntry]:
        meta = bundle.get("metadata", {})
        origin = bundle.get("origin_db", "")

        text = ""
        entry: Dict[str, Any] = {"id": bundle.get("record_id")}

        if origin == "error":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "bot":
            text = meta.get("name") or meta.get("purpose") or ""
            if "name" in meta:
                entry["name"] = meta["name"]
        elif origin == "workflow":
            text = meta.get("title") or meta.get("description") or ""
            if "title" in meta:
                entry["title"] = meta["title"]
        elif origin == "discrepancy":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "code":
            text = meta.get("summary") or meta.get("code") or ""

        entry["desc"] = self._summarise(str(text))
        metric = self._metric(origin, meta)
        if metric is not None:
            entry["metric"] = metric

        score = float(bundle.get("score", 0.0)) + (metric or 0.0)
        score *= self.db_weights.get(origin, 1.0)

        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "code": "code",
            "discrepancy": "discrepancies",
        }
        return key_map.get(origin, ""), _ScoredEntry(entry, score)

    # ------------------------------------------------------------------
    @log_and_measure
    def build_context(self, query: str, top_k: int = 5, **_: Any) -> str:
        """Return a compact JSON context for ``query``."""

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        cache_key = (query, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            hits = self.retriever.search(query, top_k=top_k * 5)
        except RateLimitError:
            raise
        except VectorServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorServiceError("retriever failure") from exc

        if isinstance(hits, ErrorResult):
            return "{}"
        if isinstance(hits, FallbackResult):
            logger.debug(
                "retriever returned fallback for %s: %s",
                query,
                getattr(hits, "reason", ""),
            )
            hits = list(hits)

        buckets: Dict[str, List[_ScoredEntry]] = {
            "errors": [],
            "bots": [],
            "workflows": [],
            "code": [],
            "discrepancies": [],
        }

        for bundle in hits:
            bucket, scored = self._bundle_to_entry(bundle)
            if bucket:
                buckets[bucket].append(scored)

        result: Dict[str, List[Dict[str, Any]]] = {}
        for key, items in buckets.items():
            if not items:
                continue
            items.sort(key=lambda e: e.score, reverse=True)
            result[key] = [e.entry for e in items[:top_k]]

        context = json.dumps(result, separators=(",", ":"))
        self._cache[cache_key] = context
        return context

    # ------------------------------------------------------------------
    @log_and_measure
    def build(self, query: str, **kwargs: Any) -> str:
        """Backward compatible alias for :meth:`build_context`.

        Older modules invoked :meth:`build` on the service layer.  The
        canonical interface is :meth:`build_context`; this wrapper simply
        forwards the call so legacy imports continue to function.
        """

        return self.build_context(query, **kwargs)

    # ------------------------------------------------------------------
    @log_and_measure
    async def build_async(self, query: str, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`build_context`."""

        return await asyncio.to_thread(self.build_context, query, **kwargs)


__all__ = ["ContextBuilder"]

