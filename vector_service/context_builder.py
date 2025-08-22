"""Cross-database context builder used by language model prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
import uuid
import time

from redaction_utils import redact_text

from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError
from .retriever import Retriever, FallbackResult
from config import ContextBuilderConfig
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from .patch_logger import _VECTOR_RISK  # type: ignore

try:  # pragma: no cover - optional precise tokenizer
    import tiktoken

    _FALLBACK_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - dependency missing or failed
    tiktoken = None  # type: ignore
    _FALLBACK_ENCODER = None

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())

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
    metadata: Dict[str, Any]


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
        safety_weight: float = getattr(ContextBuilderConfig(), "safety_weight", 1.0),
        max_tokens: int = ContextBuilderConfig().max_tokens,
        regret_penalty: float = getattr(ContextBuilderConfig(), "regret_penalty", 1.0),
        alignment_penalty: float = getattr(ContextBuilderConfig(), "alignment_penalty", 1.0),
        alert_penalty: float = getattr(ContextBuilderConfig(), "alert_penalty", 1.0),
        risk_penalty: float = getattr(ContextBuilderConfig(), "risk_penalty", 1.0),
        max_alignment_severity: float = getattr(
            ContextBuilderConfig(), "max_alignment_severity", 1.0
        ),
        max_alerts: int = getattr(ContextBuilderConfig(), "max_alerts", 5),
        license_denylist: set[str] | None = getattr(
            ContextBuilderConfig(), "license_denylist", _DEFAULT_LICENSE_DENYLIST
        ),
        precise_token_count: bool = getattr(
            ContextBuilderConfig(), "precise_token_count", True
        ),
    ) -> None:
        self.retriever = retriever or Retriever()

        if ranking_model is None:
            try:  # pragma: no cover - best effort model load
                from pathlib import Path

                try:  # package relative import when available
                    from .. import retrieval_ranker as _rr  # type: ignore
                except Exception:  # pragma: no cover - fallback
                    import retrieval_ranker as _rr  # type: ignore

                cfg = Path("retrieval_ranker.json")
                model_path = cfg
                if cfg.exists():
                    try:
                        data = json.loads(cfg.read_text())
                        if isinstance(data, dict) and data.get("current"):
                            model_path = Path(str(data["current"]))
                    except Exception:
                        pass
                self.ranking_model = _rr.load_model(model_path)
            except Exception:
                self.ranking_model = None
        else:
            self.ranking_model = ranking_model
        self.roi_tracker = roi_tracker
        self.ranking_weight = ranking_weight
        self.roi_weight = roi_weight
        self.safety_weight = safety_weight
        self.regret_penalty = regret_penalty
        self.alignment_penalty = alignment_penalty
        self.alert_penalty = alert_penalty
        self.risk_penalty = risk_penalty
        self.max_alignment_severity = max_alignment_severity
        self.max_alerts = max_alerts
        self.license_denylist = set(license_denylist or ())
        self.memory = memory_manager
        self._cache: Dict[Tuple[str, int], str] = {}
        self.db_weights = db_weights or {}
        if not self.db_weights:
            try:
                self.refresh_db_weights()
            except Exception:
                pass
        self.max_tokens = max_tokens
        self.precise_token_count = precise_token_count

        # Attempt to use tokenizer from retriever or embedder if provided.
        tok = getattr(self.retriever, "tokenizer", None)
        if tok is None:
            tok = getattr(getattr(self.retriever, "embedder", None), "tokenizer", None)
        self._tokenizer = tok
        self._fallback_tokenizer = (
            _FALLBACK_ENCODER if self.precise_token_count else None
        )

        # propagate thresholds to retriever when possible
        try:
            self.retriever.max_alert_severity = max_alignment_severity
            self.retriever.max_alerts = max_alerts
            self.retriever.license_denylist = self.license_denylist
        except Exception:
            pass

    # ------------------------------------------------------------------
    def refresh_db_weights(
        self,
        weights: Dict[str, float] | None = None,
        *,
        vector_metrics: "VectorMetricsDB" | None = None,
    ) -> None:
        """Refresh ranking weights for origin databases.

        Parameters
        ----------
        weights:
            Optional mapping of database name to weight. When omitted the
            method attempts to load weights from ``vector_metrics`` or the
            global :class:`VectorMetricsDB` instance.
        vector_metrics:
            Database from which weights are loaded when ``weights`` is ``None``.
            The argument defaults to the module-level instance when available.
        """

        global _VEC_METRICS
        if weights is None:
            vm = vector_metrics or _VEC_METRICS
            if vm is None:
                return
            try:
                weights = vm.get_db_weights()
            except Exception:
                return
            if vector_metrics is not None:
                _VEC_METRICS = vector_metrics
        if not isinstance(weights, dict):  # pragma: no cover - defensive
            return
        # Replace existing mapping so each refresh reflects the latest weights
        # from the metrics database.  This avoids stale entries lingering after
        # patches adjust the ranking model.
        try:
            self.db_weights.clear()
            self.db_weights.update(weights)
        except Exception:  # pragma: no cover - best effort
            self.db_weights = dict(weights)

    # ------------------------------------------------------------------
    def _summarise(self, text: str) -> str:
        if self.memory and hasattr(self.memory, "_summarise_text"):
            try:
                return self.memory._summarise_text(text)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - fallback
                pass
        return _summarise_text(text)

    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Return an estimate of tokens for ``text``.

        The method prefers a tokenizer supplied by the retriever or its
        embedder.  When unavailable, and ``precise_token_count`` is enabled, it
        attempts to use a lightweight dependency such as ``tiktoken`` for more
        accurate measurement.  If this dependency is missing or disabled the
        method falls back to a regex approximation which counts contiguous word
        characters.  The goal here is not perfect parity with any model but a
        consistent budget estimate for trimming.
        """
        if self._tokenizer is not None:
            try:  # pragma: no cover - defensive against tokeniser failures
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        if self.precise_token_count and self._fallback_tokenizer is not None:
            try:  # pragma: no cover - fallback tokenizer
                return len(self._fallback_tokenizer.encode(text))
            except Exception:
                pass
        return len(re.findall(r"\w+", text))

    # ------------------------------------------------------------------
    def _metric(
        self,
        origin: str,
        meta: Dict[str, Any],
        query: str,
        text: str,
        vector_id: str,
    ) -> float | None:
        """Extract ROI/success metrics from metadata and ranking model.

        The method merges metadata from the retriever with historical safety
        signals stored in :class:`VectorMetricsDB`.  When a vector ID is
        provided, win/regret rates and alignment severity are looked up and
        merged into *meta* so downstream consumers can surface them.
        """

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
            elif origin == "information":
                for key in ("roi", "data_depth", "data_depth_score", "quality"):
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

        # Patch safety metrics supplied by PatchLogger or VectorMetricsDB
        win_rate = meta.get("win_rate")
        regret_rate = meta.get("regret_rate")
        sev = meta.get("alignment_severity")

        if _VEC_METRICS is not None and vector_id:
            try:  # pragma: no cover - best effort lookup
                if win_rate is None or regret_rate is None:
                    cur = _VEC_METRICS.conn.execute(
                        "SELECT AVG(win), AVG(regret) FROM vector_metrics WHERE vector_id=?",
                        (vector_id,),
                    )
                    row = cur.fetchone()
                    if win_rate is None and row and row[0] is not None:
                        win_rate = float(row[0])
                        meta["win_rate"] = win_rate
                    if regret_rate is None and row and row[1] is not None:
                        regret_rate = float(row[1])
                        meta["regret_rate"] = regret_rate
                if sev is None:
                    cur = _VEC_METRICS.conn.execute(
                        "SELECT MAX(alignment_severity) FROM patch_ancestry WHERE vector_id=?",
                        (vector_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        sev = float(row[0])
                        meta["alignment_severity"] = sev
            except Exception:
                pass

        alerts = meta.get("semantic_alerts")
        try:
            if win_rate is not None or regret_rate is not None:
                win = min(max(float(win_rate or 0.0), 0.0), 1.0)
                regret = min(max(float(regret_rate or 0.0), 0.0), 1.0)
                metric = (metric or 0.0) + self.safety_weight * (win - regret)
            if alerts:
                metric = (metric or 0.0) - (
                    float(len(alerts))
                    if isinstance(alerts, (list, tuple, set))
                    else 1.0
                )
        except Exception:  # pragma: no cover - defensive
            pass

        if sev is not None:
            try:
                sev_val = min(max(float(sev), 0.0), 1.0)
                metric = (metric or 0.0) - self.safety_weight * sev_val
            except Exception:
                pass

        lic = meta.get("license")
        fp = meta.get("license_fingerprint")
        if lic in self.license_denylist or _LICENSE_DENYLIST.get(fp) in self.license_denylist:
            metric = (metric or 0.0) - self.safety_weight

        if self.ranking_model is not None:
            try:
                if hasattr(self.ranking_model, "score"):
                    metric = (metric or 0.0) + float(
                        self.ranking_model.score(query, text)
                    )
                elif hasattr(self.ranking_model, "rank"):
                    metric = (metric or 0.0) + float(
                        self.ranking_model.rank(query, text)
                    )
                else:
                    metric = (metric or 0.0) + float(
                        self.ranking_model(query, text)  # type: ignore[misc]
                    )
            except Exception:
                pass

        return metric

    # ------------------------------------------------------------------
    def _bundle_to_entry(self, bundle: Dict[str, Any], query: str) -> Tuple[str, _ScoredEntry]:
        meta = bundle.get("metadata", {})
        origin = bundle.get("origin_db", "")

        text = bundle.get("text") or ""
        vec_id = str(bundle.get("record_id", ""))
        alerts = bundle.get("semantic_alerts") or meta.get("semantic_alerts")
        severity = bundle.get("alignment_severity") or meta.get("alignment_severity")
        try:
            if (
                severity is not None
                and float(severity) > self.max_alignment_severity
            ) or (
                alerts is not None
                and (
                    (len(alerts) if isinstance(alerts, (list, tuple, set)) else 1)
                    > self.max_alerts
                )
            ):
                if _VECTOR_RISK is not None:
                    _VECTOR_RISK.labels("filtered").inc()
                return "", _ScoredEntry({}, 0.0, origin, vec_id, {})
        except Exception:
            pass
        lic = bundle.get("license") or meta.get("license")
        fp = bundle.get("license_fingerprint") or meta.get("license_fingerprint")
        if lic in self.license_denylist or _LICENSE_DENYLIST.get(fp) in self.license_denylist:
            if _VECTOR_RISK is not None:
                _VECTOR_RISK.labels("filtered").inc()
            return "", _ScoredEntry({}, 0.0, origin, vec_id, {})
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
            text = (
                text
                or meta.get("title")
                or meta.get("description")
                or meta.get("lessons")
                or ""
            )
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
            elif "name" in meta:
                entry["name"] = redact_text(str(meta["name"]))
            if meta.get("lessons"):
                entry["lessons"] = self._summarise(
                    redact_text(str(meta["lessons"]))
                )
        elif origin == "information":
            text = (
                text
                or meta.get("title")
                or meta.get("summary")
                or meta.get("content")
                or meta.get("lessons")
                or ""
            )
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
            elif "name" in meta:
                entry["name"] = redact_text(str(meta["name"]))
            if meta.get("lessons"):
                entry["lessons"] = self._summarise(
                    redact_text(str(meta["lessons"]))
                )
        elif origin == "discrepancy":
            text = text or meta.get("message") or meta.get("description") or ""
        elif origin == "code":
            text = text or meta.get("summary") or meta.get("code") or ""

        text = redact_text(str(text))
        entry["desc"] = text
        metric = self._metric(origin, meta, query, text, vec_id)
        if metric is not None:
            entry["metric"] = metric

        roi_val = meta.get("roi") if isinstance(meta, dict) else None
        if roi_val is None:
            roi_val = bundle.get("roi")
        if roi_val is not None:
            try:
                entry["roi"] = float(roi_val)
            except Exception:
                pass

        risk_val = None
        for key in ("risk_score", "final_risk_score", "risk"):
            if isinstance(meta, dict) and meta.get(key) is not None:
                risk_val = meta.get(key)
                break
            if bundle.get(key) is not None:
                risk_val = bundle.get(key)
                break
        if risk_val is not None:
            try:
                risk_val = float(risk_val)
                entry["risk_score"] = risk_val
            except Exception:
                risk_val = None

        # Surface patch safety metrics when available
        win_rate = meta.get("win_rate")
        regret_rate = meta.get("regret_rate")
        if win_rate is not None:
            try:
                entry["win_rate"] = float(win_rate)
            except Exception:
                pass
        if regret_rate is not None:
            try:
                entry["regret_rate"] = float(regret_rate)
            except Exception:
                pass

        # Patch safety flags
        flags: Dict[str, Any] = {}
        lic = bundle.get("license") or meta.get("license")
        fp = bundle.get("license_fingerprint") or meta.get("license_fingerprint")
        if lic:
            flags["license"] = lic
        if fp:
            flags["license_fingerprint"] = fp
        if alerts:
            flags["semantic_alerts"] = alerts
        if severity is not None:
            flags["alignment_severity"] = severity
        if flags:
            entry["flags"] = flags

        if _VEC_METRICS is not None and origin:
            try:  # pragma: no cover - best effort metrics lookup
                entry.setdefault(
                    "win_rate", _VEC_METRICS.retriever_win_rate(origin)
                )
                entry.setdefault(
                    "regret_rate", _VEC_METRICS.retriever_regret_rate(origin)
                )
            except Exception:
                pass

        penalty = 0.0
        if regret_rate is not None:
            try:
                penalty += float(regret_rate) * self.regret_penalty
            except Exception:
                pass
        if severity is not None:
            try:
                penalty += float(severity) * self.alignment_penalty
            except Exception:
                pass
        if risk_val is not None:
            try:
                penalty += float(risk_val) * self.risk_penalty
            except Exception:
                pass
        if alerts:
            penalty += (
                len(alerts) if isinstance(alerts, (list, tuple, set)) else 1.0
            ) * self.alert_penalty
        if lic in self.license_denylist or _LICENSE_DENYLIST.get(fp) in self.license_denylist:
            penalty += 1.0
        penalty *= self.safety_weight

        similarity = float(bundle.get("score", 0.0))
        rank_prob = self.ranking_weight
        roi_bias = self.roi_weight
        if self.roi_tracker is not None:
            try:
                roi_bias = float(
                    self.roi_tracker.retrieval_bias().get(origin, self.roi_weight)
                )
            except Exception:
                roi_bias = self.roi_weight

        score = similarity * rank_prob * roi_bias + (metric or 0.0) - penalty
        score *= self.db_weights.get(origin, 1.0)

        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "enhancement": "enhancements",
            "information": "information",
            "code": "code",
            "discrepancy": "discrepancies",
        }
        return key_map.get(origin, ""), _ScoredEntry(entry, score, origin, vec_id, meta)

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
        return_stats: bool = False,
        prioritise: str | None = None,
        **_: Any,
    ) -> Any:
        """Return a compact JSON context for ``query``.

        Parameters
        ----------
        query:
            Search query used when retrieving vectors.
        top_k:
            Maximum number of entries from each bucket to consider before
            trimming.
        include_vectors:
            When ``True`` the return value includes vector IDs and scores.
        return_metadata:
            When ``True`` the full metadata for each entry is returned.
        prioritise:
            Optional trimming strategy. ``"newest"`` prefers more recent
            entries while ``"roi"`` favours higher ROI vectors.

        When ``include_vectors`` is True, the return value is a tuple of
        ``(context_json, session_id, vectors)`` where *vectors* is a list of
        ``(origin, vector_id, score)`` triples.  If ``return_metadata`` is
        enabled, the metadata dictionary is appended as the final element of the
        tuple and contains the full entries including reliability metrics and
        safety flags.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        try:
            self.refresh_db_weights()
        except Exception:
            pass

        prompt_tokens = len(query.split())
        query = redact_text(query)
        cache_key = (query, top_k)
        if not include_vectors and not return_metadata and cache_key in self._cache:
            return self._cache[cache_key]

        session_id = session_id or uuid.uuid4().hex
        start = time.perf_counter()
        try:
            hits = self.retriever.search(
                query,
                top_k=top_k * 5,
                session_id=session_id,
                max_alert_severity=self.max_alignment_severity,
            )
        except RateLimitError:
            raise
        except VectorServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorServiceError("retriever failure") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

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
            "information": [],
            "code": [],
            "discrepancies": [],
        }

        for bundle in hits:
            bucket, scored = self._bundle_to_entry(bundle, query)
            if bucket:
                buckets[bucket].append(scored)

        # Flatten scored entries and compute token estimates so we can trim
        # globally across buckets.
        bucket_order = list(buckets.keys())
        candidates: List[Dict[str, Any]] = []
        for key in bucket_order:
            items = buckets[key]
            if not items:
                continue
            items.sort(key=lambda e: e.score, reverse=True)
            for e in items[:top_k]:
                full = e.entry
                summary = {
                    k: v for k, v in full.items() if k not in {"win_rate", "regret_rate", "flags"}
                }
                cand = {
                    "bucket": key,
                    "summary": summary,
                    "meta": full,
                    "raw": e.metadata,
                    "score": e.score,
                    "origin": e.origin,
                    "vector_id": e.vector_id,
                    "summarised": False,
                }
                cand["tokens"] = self._count_tokens(
                    json.dumps(summary, separators=(",", ":"))
                )
                candidates.append(cand)

        def estimate_tokens(cands: List[Dict[str, Any]]) -> int:
            ctx: Dict[str, List[Dict[str, Any]]] = {}
            for c in cands:
                ctx.setdefault(c["bucket"], []).append(c["summary"])
            return self._count_tokens(json.dumps(ctx, separators=(",", ":")))
        sum_tokens = sum(c["tokens"] for c in candidates)
        total_tokens = estimate_tokens(candidates)
        overhead = total_tokens - sum_tokens
        total_tokens = sum_tokens + overhead

        if total_tokens > self.max_tokens and candidates:
            if prioritise == "newest":
                candidates.sort(
                    key=lambda c: (
                        c["score"],
                        c["raw"].get("timestamp")
                        or c["raw"].get("ts")
                        or c["raw"].get("created_at")
                        or c["raw"].get("id", 0),
                    )
                )
            elif prioritise == "roi":
                candidates.sort(key=lambda c: (c["score"], c["raw"].get("roi", 0)))
            else:
                candidates.sort(key=lambda c: c["score"])

            idx = 0
            while total_tokens > self.max_tokens and candidates:
                cand = candidates[idx]
                desc = cand["summary"].get("desc", "")
                if not cand["summarised"]:
                    cand["summary"]["desc"] = self._summarise(desc)
                    cand["meta"]["desc"] = cand["summary"]["desc"]
                    cand["meta"]["truncated"] = True
                    cand["summarised"] = True
                    new_tokens = self._count_tokens(
                        json.dumps(cand["summary"], separators=(",", ":"))
                    )
                    sum_tokens += new_tokens - cand["tokens"]
                    cand["tokens"] = new_tokens
                    total_tokens = sum_tokens + overhead
                else:
                    truncated = desc.rsplit(" ", 1)[0] if " " in desc else ""
                    if not truncated or truncated == desc:
                        sum_tokens -= cand["tokens"]
                        candidates.pop(idx)
                        if candidates:
                            sum_tokens = sum(c["tokens"] for c in candidates)
                            overhead = estimate_tokens(candidates) - sum_tokens
                            total_tokens = sum_tokens + overhead
                        else:
                            total_tokens = 0
                            sum_tokens = 0
                            overhead = 0
                    else:
                        cand["summary"]["desc"] = truncated + "..."
                        cand["meta"]["desc"] = cand["summary"]["desc"]
                        cand["meta"]["truncated"] = True
                        new_tokens = self._count_tokens(
                            json.dumps(cand["summary"], separators=(",", ":"))
                        )
                        sum_tokens += new_tokens - cand["tokens"]
                        cand["tokens"] = new_tokens
                        total_tokens = sum_tokens + overhead

        result: Dict[str, List[Dict[str, Any]]] = {}
        meta: Dict[str, List[Dict[str, Any]]] = {}
        vectors: List[Tuple[str, str, float]] = []
        for key in bucket_order:
            for c in candidates:
                if c["bucket"] == key:
                    result.setdefault(key, []).append(c["summary"])
                    if return_metadata:
                        meta.setdefault(key, []).append(c["meta"])
                    vectors.append((c["origin"], c["vector_id"], c["score"]))

        context = json.dumps(result, separators=(",", ":"))
        total_tokens = self._count_tokens(context)
        if not include_vectors and not return_metadata:
            self._cache[cache_key] = context
        stats = {
            "tokens": total_tokens,
            "wall_time_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
        }
        if include_vectors and return_metadata:
            if return_stats:
                return context, session_id, vectors, meta, stats
            return context, session_id, vectors, meta
        if include_vectors:
            if return_stats:
                return context, session_id, vectors, stats
            return context, session_id, vectors
        if return_metadata:
            if return_stats:
                return context, meta, stats
            return context, meta
        if return_stats:
            return context, stats
        return context

    # ------------------------------------------------------------------
    @log_and_measure
    def build(self, query: str, **kwargs: Any) -> Any:
        """Backward compatible alias for :meth:`build_context`.

        Older modules invoked :meth:`build` on the service layer.  The
        canonical interface is :meth:`build_context`; this wrapper simply
        forwards the call so legacy imports continue to function.
        """

        return self.build_context(query, **kwargs)

    # ------------------------------------------------------------------
    @log_and_measure
    async def build_async(self, query: str, **kwargs: Any) -> Any:
        """Asynchronous wrapper for :meth:`build_context`."""

        return await asyncio.to_thread(self.build_context, query, **kwargs)


__all__ = ["ContextBuilder"]

