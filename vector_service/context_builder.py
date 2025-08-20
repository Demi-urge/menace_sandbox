"""Cross-database context builder used by language model prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
import uuid

from redaction_utils import redact_text

from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError
from .retriever import Retriever, FallbackResult
from config import ContextBuilderConfig

try:  # pragma: no cover - optional dependency
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

_VEC_METRICS = VectorMetricsDB() if VectorMetricsDB is not None else None

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
    origin: str
    vector_id: str


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        ranking_model: Any | None = None,
        roi_tracker: Any | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
        db_weights: Dict[str, float] | None = None,
        ranking_weight: float = ContextBuilderConfig().ranking_weight,
        roi_weight: float = ContextBuilderConfig().roi_weight,
        max_tokens: int = ContextBuilderConfig().max_tokens,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.ranking_model = ranking_model
        self.roi_tracker = roi_tracker
        self.ranking_weight = ranking_weight
        self.roi_weight = roi_weight
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

        metric: float | None = None
        try:
            if origin == "error":
                freq = meta.get("frequency")
                if freq is not None:
                    metric = 1.0 / (1.0 + float(freq))
            elif origin == "bot":
                for key in ("roi", "deploy_count"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "workflow":
                for key in ("roi", "usage", "runs"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "enhancement":
                for key in ("roi", "adoption"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "code":
                for key in ("roi", "patch_success"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "discrepancy":
                for key in ("roi", "severity", "impact"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
        except Exception:  # pragma: no cover - defensive
            metric = None

        sev = meta.get("alignment_severity")
        if sev is not None:
            try:
                metric = (metric or 0.0) - float(sev)
            except Exception:
                pass
        return metric

    # ------------------------------------------------------------------
    def _bundle_to_entry(self, bundle: Dict[str, Any], query: str) -> Tuple[str, _ScoredEntry]:
        meta = bundle.get("metadata", {})
        origin = bundle.get("origin_db", "")

        text = bundle.get("text") or ""
        entry: Dict[str, Any] = {"id": bundle.get("record_id")}

        if origin == "error":
            text = text or meta.get("message") or meta.get("description") or ""
        elif origin == "bot":
            text = text or meta.get("name") or meta.get("purpose") or ""
            if "name" in meta:
                entry["name"] = redact_text(str(meta["name"]))
        elif origin == "workflow":
            text = text or meta.get("title") or meta.get("description") or ""
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
        elif origin == "enhancement":
            text = text or meta.get("title") or meta.get("description") or ""
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
        elif origin == "discrepancy":
            text = text or meta.get("message") or meta.get("description") or ""
        elif origin == "code":
            text = text or meta.get("summary") or meta.get("code") or ""

        text = redact_text(str(text))
        entry["desc"] = self._summarise(text)
        metric = self._metric(origin, meta)
        if metric is not None:
            entry["metric"] = metric

        # Patch safety flags
        flags: Dict[str, Any] = {}
        lic = bundle.get("license") or meta.get("license")
        alerts = bundle.get("semantic_alerts") or meta.get("semantic_alerts")
        severity = bundle.get("alignment_severity") or meta.get("alignment_severity")
        if lic:
            flags["license"] = lic
        if alerts:
            flags["semantic_alerts"] = alerts
        if severity is not None:
            flags["alignment_severity"] = severity
        if flags:
            entry["flags"] = flags

        if _VEC_METRICS is not None and origin:
            try:  # pragma: no cover - best effort metrics lookup
                entry["win_rate"] = _VEC_METRICS.retriever_win_rate(origin)
                entry["regret_rate"] = _VEC_METRICS.retriever_regret_rate(origin)
            except Exception:
                pass

        similarity = float(bundle.get("score", 0.0))
        rank_prob = self.ranking_weight
        if self.ranking_model is not None:
            try:
                if hasattr(self.ranking_model, "score"):
                    rank_prob = float(self.ranking_model.score(query, text))
                elif hasattr(self.ranking_model, "rank"):
                    rank_prob = float(self.ranking_model.rank(query, text))
                else:
                    rank_prob = float(self.ranking_model(query, text))  # type: ignore[misc]
            except Exception:
                rank_prob = self.ranking_weight
        roi_bias = self.roi_weight
        if self.roi_tracker is not None:
            try:
                roi_bias = float(
                    self.roi_tracker.retrieval_bias().get(origin, self.roi_weight)
                )
            except Exception:
                roi_bias = self.roi_weight

        score = similarity * rank_prob * roi_bias + (metric or 0.0)
        score *= self.db_weights.get(origin, 1.0)

        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "enhancement": "enhancements",
            "code": "code",
            "discrepancy": "discrepancies",
        }
        return key_map.get(origin, ""), _ScoredEntry(entry, score, origin, str(bundle.get("record_id", "")))

    # ------------------------------------------------------------------
    @log_and_measure
    def build_context(
        self,
        query: str,
        top_k: int = 5,
        *,
        include_vectors: bool = False,
        return_metadata: bool = False,
        session_id: str | None = None,
        **_: Any,
    ) -> str | Tuple[str, str, List[Tuple[str, str, float]]] | Tuple[str, Dict[str, List[Dict[str, Any]]]] | Tuple[str, str, List[Tuple[str, str, float]], Dict[str, List[Dict[str, Any]]]]:
        """Return a compact JSON context for ``query``.

        When ``include_vectors`` is True, the return value is a tuple of
        ``(context_json, session_id, vectors)`` where *vectors* is a list of
        ``(origin, vector_id, score)`` triples.  If ``return_metadata`` is
        enabled, the metadata dictionary is appended as the final element of the
        tuple and contains the full entries including reliability metrics and
        safety flags.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        query = redact_text(query)
        cache_key = (query, top_k)
        if not include_vectors and not return_metadata and cache_key in self._cache:
            return self._cache[cache_key]

        session_id = session_id or uuid.uuid4().hex
        try:
            hits = self.retriever.search(query, top_k=top_k * 5, session_id=session_id)
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
            "enhancements": [],
            "code": [],
            "discrepancies": [],
        }

        for bundle in hits:
            bucket, scored = self._bundle_to_entry(bundle, query)
            if bucket:
                buckets[bucket].append(scored)

        result: Dict[str, List[Dict[str, Any]]] = {}
        meta: Dict[str, List[Dict[str, Any]]] = {}
        vectors: List[Tuple[str, str, float]] = []
        for key, items in buckets.items():
            if not items:
                continue
            items.sort(key=lambda e: e.score, reverse=True)
            chosen = items[:top_k]
            summaries: List[Dict[str, Any]] = []
            for e in chosen:
                full = e.entry
                if return_metadata:
                    meta.setdefault(key, []).append(full)
                summaries.append({k: v for k, v in full.items() if k not in {"win_rate", "regret_rate", "flags"}})
            result[key] = summaries
            vectors.extend([(e.origin, e.vector_id, e.score) for e in chosen])

        context = json.dumps(result, separators=(",", ":"))
        if not include_vectors and not return_metadata:
            self._cache[cache_key] = context
        if include_vectors and return_metadata:
            return context, session_id, vectors, meta
        if include_vectors:
            return context, session_id, vectors
        if return_metadata:
            return context, meta
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

