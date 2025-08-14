from __future__ import annotations

import json
import sqlite3
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


@dataclass
class RetrievedItem:
    """Structured result returned by :class:`UniversalRetriever`.

    The dataclass bundles the origin of the hit, its primary key within
    the source database, the raw metadata for that record and a computed
    confidence score.  A short human readable ``reason`` explains which
    metric most influenced the ranking so downstream components can
    surface richer explanations to users.
    """

    origin_db: str
    record: Any
    confidence: float
    metadata: dict[str, Any]
    reason: str

    # Backwards compatibility for older attribute names
    @property
    def source_db(self) -> str:  # pragma: no cover - simple alias
        return self.origin_db

    @property
    def confidence_score(self) -> float:  # pragma: no cover - simple alias
        return self.confidence

    @property
    def record_id(self) -> Any:  # pragma: no cover - compatibility shim
        """Best-effort extraction of an identifier for ``record``.

        Many legacy callers expect ``RetrievedItem`` to expose ``record_id``
        even though the dataclass now stores the full ``record`` object
        instead.  We attempt to pull a likely identifier from ``record`` or
        ``metadata`` using common attribute names.  ``None`` is returned when
        no obvious identifier can be found.
        """

        if isinstance(self.record, dict):
            for key in ("id", "wid", "bid", "record_id", "info_id", "item_id"):
                if key in self.record:
                    return self.record[key]
        else:
            for key in ("id", "wid", "bid", "record_id", "info_id", "item_id"):
                if hasattr(self.record, key):
                    return getattr(self.record, key)
        if isinstance(self.metadata, dict):
            for key in ("id", "wid", "bid", "record_id", "info_id", "item_id"):
                if key in self.metadata:
                    return self.metadata[key]
        return None


# Older name retained for compatibility with existing imports
RetrievalHit = RetrievedItem


def boost_linked_candidates(
    scored: List[dict[str, Any]],
    *,
    bot_db: Any | None = None,
    error_db: Any | None = None,
    multiplier: float = 1.1,
) -> dict[int, str]:
    """Apply a score boost for items linked via bot relationships.

    ``scored`` is modified in-place. Any candidates that share a bot ID
    through the ``bot_workflow``, ``bot_error`` or ``bot_enhancement``
    tables receive the ``multiplier`` on their ``confidence`` score.  The
    function returns a mapping of candidate index to a textual linkage path
    describing how results are connected (e.g. ``"bot->workflow->error"``).
    """

    if multiplier <= 1.0 or not scored:
        return {}

    bot_sets: List[set[int]] = []
    type_map: List[str | None] = []
    for entry in scored:
        bots: set[int] = set()
        typ = entry.get("source")
        cid = entry.get("record_id")

        if typ == "bot" and cid:
            bots.add(int(cid))
        elif typ == "workflow" and cid and bot_db:
            rows = bot_db.conn.execute(
                "SELECT bot_id FROM bot_workflow WHERE workflow_id=?",
                (int(cid),),
            ).fetchall()
            bots.update(int(r[0]) for r in rows)
        elif typ == "error" and cid and error_db:
            rows = error_db.conn.execute(
                "SELECT bot_id FROM bot_error WHERE error_id=?",
                (int(cid),),
            ).fetchall()
            bots.update(int(r[0]) for r in rows)
        elif typ == "enhancement" and cid and bot_db:
            rows = bot_db.conn.execute(
                "SELECT bot_id FROM bot_enhancement WHERE enhancement_id=?",
                (int(cid),),
            ).fetchall()
            bots.update(int(r[0]) for r in rows)
        bot_sets.append(bots)
        type_map.append(typ)

    n = len(scored)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if bot_sets[i] and bot_sets[j] and bot_sets[i].intersection(bot_sets[j]):
                union(i, j)

    groups: dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    link_paths: dict[int, str] = {}
    order = ["bot", "workflow", "enhancement", "error", "information"]
    for members in groups.values():
        if len(members) > 1:
            types = {type_map[m] for m in members if type_map[m]}
            if "bot" in types:
                path_types = ["bot"] + [t for t in order[1:] if t in types]
            else:
                others = [t for t in order if t in types]
                if others:
                    first, rest = others[0], others[1:]
                    path_types = [first, "bot"] + rest
                else:
                    path_types = ["bot"]
            path_str = "->".join(path_types)
            for m in members:
                if "confidence" in scored[m]:
                    scored[m]["confidence"] *= multiplier
                link_paths[m] = path_str

    return link_paths


class UniversalRetriever:
    """Search across multiple databases using vector similarity."""

    def __init__(
        self,
        *,
        bot_db: Any | None = None,
        workflow_db: Any | None = None,
        error_db: Any | None = None,
        enhancement_db: Any | None = None,
        information_db: Any | None = None,
        knowledge_graph: Any | None = None,
    ) -> None:
        self.bot_db = bot_db
        self.workflow_db = workflow_db
        self.error_db = error_db
        self.enhancement_db = enhancement_db
        self.information_db = information_db
        self.graph = knowledge_graph
        self._deploy_conn: sqlite3.Connection | None = None

        self._encoder = next(
            (
                db
                for db in (
                    bot_db,
                    workflow_db,
                    error_db,
                    enhancement_db,
                    information_db,
                )
                if db is not None
            ),
            None,
        )
        if self._encoder is None:
            raise ValueError("At least one database instance is required")

    # ------------------------------------------------------------------
    def _extract_id(self, obj: Any, names: Sequence[str]) -> Any | None:
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return None
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _to_vector(self, query: Any) -> List[float]:
        """Convert ``query`` into an embedding vector.

        Parameters
        ----------
        query:
            Either raw text, a record/record ID from one of the configured
            databases or an explicit numeric vector.  Strings are encoded
            using the first available database's ``encode_text`` method.  For
            record objects we first try the database's ``vector`` method and
            fall back to ``get_vector`` using common identifier fields.  If
            ``query`` is already a sequence of numbers it is returned as-is.
        """

        if isinstance(query, str):
            return self._encoder.encode_text(query)

        if isinstance(query, Sequence) and not isinstance(query, (bytes, bytearray)):
            try:
                return [float(x) for x in query]
            except Exception:  # pragma: no cover - defensive
                pass

        mapping = [
            (self.bot_db, ("id", "bid")),
            (self.workflow_db, ("id", "wid")),
            (self.error_db, ("id",)),
            (self.enhancement_db, ("id",)),
            (self.information_db, ("id", "info_id", "item_id")),
        ]

        for db, names in mapping:
            if db is None:
                continue
            # Try direct vectorisation of the record instance
            try:
                vec = db.vector(query)
                if vec is not None:
                    return list(vec)
            except Exception:
                pass
            # Fallback to stored vector via identifier lookup
            rec_id = self._extract_id(query, names)
            if rec_id is not None:
                try:
                    vec = db.get_vector(rec_id)
                    if vec is not None:
                        return list(vec)
                except Exception:  # pragma: no cover - defensive
                    continue

        raise TypeError("Unsupported query type for retrieval")

    # ------------------------------------------------------------------
    def _retrieve_candidates(
        self, query: Any, top_k: int = 10
    ) -> List[tuple[str, Any, Any]]:
        """Return raw candidates and their sources for ``query``."""

        vector = self._to_vector(query)
        candidates: List[tuple[str, Any, Any]] = []

        mapping = [
            ("bot", self.bot_db, ("id", "bid")),
            ("workflow", self.workflow_db, ("id", "wid")),
            ("error", self.error_db, ("id",)),
            ("enhancement", self.enhancement_db, ("id",)),
            ("information", self.information_db, ("id", "info_id")),
        ]

        for source, db, names in mapping:
            if db is None:
                continue
            try:
                matches = db.search_by_vector(vector, top_k)
            except Exception:  # pragma: no cover - defensive
                continue
            for m in matches:
                rec_id = self._extract_id(m, names)
                candidates.append((source, rec_id, m))

        def _dist(entry: tuple[str, Any, Any]) -> float:
            item = entry[2]
            if isinstance(item, dict):
                return float(item.get("_distance", float("inf")))
            return float(getattr(item, "_distance", float("inf")))

        candidates.sort(key=_dist)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    def _error_frequency(self, error_id: int) -> float:
        """Return raw error frequency from ``ErrorDB``."""

        if not self.error_db:
            return 0.0
        try:
            cur = self.error_db.conn.execute(
                "SELECT frequency FROM errors WHERE id=?",
                (error_id,),
            ).fetchone()
            if cur and cur[0] is not None:
                return float(cur[0])
        except sqlite3.Error:
            return 0.0
        return 0.0

    def _enhancement_roi(self, enh: Any) -> float:
        """Return ROI uplift score for an enhancement record."""

        if not self.enhancement_db:
            return 0.0
        try:
            return float(getattr(enh, "score", 0.0))
        except Exception:
            return 0.0

    def _workflow_usage(self, wf: Any) -> float:
        """Return approximate usage count for a workflow record."""

        try:
            data = json.loads(getattr(wf, "performance_data", "") or "{}")
            for key in ("runs", "executions", "usage", "count"):
                if key in data:
                    return float(data[key])
        except Exception:
            pass
        assigned = getattr(wf, "assigned_bots", []) or []
        return float(len(assigned))

    def _bot_deploy_freq(self, bot_id: int) -> float:
        """Return deployment count for a bot from deployment logs."""

        if not self.bot_db:
            return 0.0
        if getattr(self, "_deploy_conn", None) is None:
            try:
                self._deploy_conn = sqlite3.connect("deployment.db")
            except sqlite3.Error:
                return 0.0
        try:
            cur = self._deploy_conn.execute(
                "SELECT COUNT(*) FROM bot_trials WHERE bot_id=?",
                (bot_id,),
            ).fetchone()
            if cur and cur[0] is not None:
                return float(cur[0])
        except sqlite3.Error:
            return 0.0
        return 0.0

    # ------------------------------------------------------------------
    def _context_score(self, kind: str, record: Any) -> tuple[float, dict[str, float]]:
        """Compute contextual score and raw metrics for a record.

        Parameters
        ----------
        kind:
            Source type such as ``"error"`` or ``"workflow"``.
        record:
            The record object returned from the respective database.

        Returns
        -------
        tuple
            A ``(score, metrics)`` pair where ``score`` is a normalised
            value between 0 and 1 and ``metrics`` contains the raw metric
            values used.
        """

        metrics: dict[str, float] = {}
        score = 0.0
        if kind == "error":
            err_id = self._extract_id(record, ("id",))
            if err_id is not None:
                freq = self._error_frequency(int(err_id))
                metrics["frequency"] = freq
                lf = math.log1p(freq)
                score = lf / (1.0 + lf)
        elif kind == "enhancement":
            roi = self._enhancement_roi(record)
            metrics["roi"] = roi
            roi_pos = max(roi, 0.0)
            score = roi_pos / (1.0 + roi_pos)
        elif kind == "workflow":
            usage = self._workflow_usage(record)
            metrics["usage"] = usage
            lu = math.log1p(usage)
            score = lu / (1.0 + lu)
        elif kind == "bot":
            bid = self._extract_id(record, ("id", "bid"))
            if bid is not None:
                freq = self._bot_deploy_freq(int(bid))
                metrics["deploy"] = freq
                lf = math.log1p(freq)
                score = lf / (1.0 + lf)
        return score, metrics

    def retrieve(
        self, query: Any, top_k: int = 10, link_multiplier: float = 1.1
    ) -> List[RetrievedItem]:
        """Retrieve results with confidence scores and reasons."""

        raw_results = self._retrieve_candidates(query, top_k)

        SIM_WEIGHT = 0.7
        CTX_WEIGHT = 0.3

        scored: List[dict[str, Any]] = []
        for source, rec_id, item in raw_results:
            dist = item["_distance"] if isinstance(item, dict) else getattr(item, "_distance", 0.0)
            similarity = 1.0 / (1.0 + float(dist))
            ctx_score, metrics = self._context_score(source, item)
            combined_score = similarity * SIM_WEIGHT + ctx_score * CTX_WEIGHT
            scored.append({
                "source": source,
                "record_id": rec_id,
                "item": item,
                "confidence": combined_score,
                "similarity": similarity,
                "context": ctx_score,
                **metrics,
            })

        link_paths = boost_linked_candidates(
            scored,
            bot_db=self.bot_db,
            error_db=self.error_db,
            multiplier=link_multiplier,
        )

        reason_map = {
            "roi": "high ROI uplift",
            "frequency": "frequent error",
            "usage": "heavy usage",
            "deploy": "widely deployed bot",
        }

        hits: List[RetrievedItem] = []
        for idx, entry in enumerate(scored):
            item = entry["item"]
            base_meta = item if isinstance(item, dict) else item.__dict__
            meta = dict(base_meta)
            metrics = {
                k: v
                for k, v in entry.items()
                if k not in {"source", "record_id", "item", "confidence"}
            }
            meta.update(metrics)
            metrics_for_reason = {
                k: v for k, v in metrics.items() if k not in {"similarity", "context"}
            }
            top_metric = max(metrics_for_reason, key=metrics_for_reason.get, default=None)
            reason = reason_map.get(top_metric, "relevant match")
            if top_metric:
                reason += f" ({top_metric}={metrics_for_reason[top_metric]:.2f})"
            if idx in link_paths:
                reason += f" linked via {link_paths[idx]}"
            hits.append(
                RetrievedItem(
                    origin_db=entry["source"],
                    record=item,
                    confidence=entry["confidence"],
                    metadata=meta,
                    reason=reason,
                )
            )

        hits.sort(key=lambda h: h.confidence, reverse=True)
        return hits[:top_k]

    # Backwards compatibility for older callers
    def retrieve_with_confidence(
        self, query: Any, top_k: int = 10, link_multiplier: float = 1.1
    ) -> List[dict[str, Any]]:
        hits = self.retrieve(query, top_k=top_k, link_multiplier=link_multiplier)
        return [
            {
                "source": h.origin_db,
                "record_id": h.record_id,
                "item": h.metadata,
                "confidence": h.confidence,
                "reason": h.reason,
            }
            for h in hits
        ]
