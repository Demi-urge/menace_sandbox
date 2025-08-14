from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


@dataclass
class RetrievalHit:
    """Structured result returned by :class:`UniversalRetriever`.

    The dataclass bundles the origin of the hit, its primary key within
    the source database, the raw metadata for that record and a computed
    confidence score.  A short human readable ``reason`` explains which
    metric most influenced the ranking so downstream components can
    surface richer explanations to users.
    """

    source_db: str
    record_id: Any
    metadata: dict[str, Any]
    confidence_score: float
    reason: str


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
    ) -> None:
        self.bot_db = bot_db
        self.workflow_db = workflow_db
        self.error_db = error_db
        self.enhancement_db = enhancement_db
        self.information_db = information_db

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

    def _vector_for_query(self, query: Any) -> List[float]:
        """Return an embedding vector for ``query``.

        ``query`` may be raw text or a database record.  Text queries are
        embedded using :meth:`EmbeddableDBMixin.encode_text` from whichever
        database instance was provided on construction.  For record objects
        we look up the stored vector via :meth:`EmbeddableDBMixin.get_vector`.
        """

        if isinstance(query, str):
            return self._encoder.encode_text(query)

        mapping = [
            (self.bot_db, ("id", "bid")),
            (self.workflow_db, ("id", "wid")),
            (self.error_db, ("id",)),
            (self.enhancement_db, ("id",)),
            (self.information_db, ("id", "info_id")),
        ]
        for db, names in mapping:
            if db is None:
                continue
            rec_id = self._extract_id(query, names)
            if rec_id is not None:
                vec = db.get_vector(rec_id)
                if vec is not None:
                    return vec
        raise TypeError("Unsupported query type for retrieval")

    # ------------------------------------------------------------------
    def _retrieve_candidates(
        self, query: Any, top_k: int = 10
    ) -> List[tuple[str, Any, Any]]:
        """Return raw candidates and their sources for ``query``."""

        vector = self._vector_for_query(query)
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
        if not self.error_db:
            return 0.0
        try:
            cur = self.error_db.conn.execute(
                "SELECT frequency FROM telemetry WHERE id=?", (error_id,)
            ).fetchone()
            if cur and cur[0] is not None:
                return float(cur[0])
            cur = self.error_db.conn.execute(
                "SELECT frequency FROM errors WHERE id=?", (error_id,)
            ).fetchone()
            if cur and cur[0] is not None:
                return float(cur[0])
        except sqlite3.Error:
            return 0.0
        return 0.0

    def _enhancement_roi(self, enh: Any) -> float:
        if not self.enhancement_db:
            return 0.0
        try:
            h = hashlib.sha1((getattr(enh, "after_code", "") or "").encode()).hexdigest()
            cur = self.enhancement_db.conn.execute(
                "SELECT metric_delta FROM enhancement_history WHERE enhanced_hash=? ORDER BY id DESC LIMIT 1",
                (h,),
            ).fetchone()
            if cur and cur[0] is not None:
                return float(cur[0])
        except sqlite3.Error:
            return 0.0
        return float(getattr(enh, "score", 0.0))

    def _workflow_usage(self, wf: Any) -> float:
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
        if not self.bot_db:
            return 0.0
        total = 0
        try:
            cur = self.bot_db.conn.execute(
                "SELECT COUNT(*) FROM bot_workflow WHERE bot_id=?", (bot_id,)
            ).fetchone()
            total += int(cur[0]) if cur else 0
            cur = self.bot_db.conn.execute(
                "SELECT COUNT(*) FROM bot_enhancement WHERE bot_id=?", (bot_id,)
            ).fetchone()
            total += int(cur[0]) if cur else 0
        except sqlite3.Error:
            return float(total)
        return float(total)

    def retrieve(
        self, query: Any, top_k: int = 10, link_multiplier: float = 1.1
    ) -> List[RetrievalHit]:
        """Retrieve results with confidence scores and reasons."""

        raw_results = self._retrieve_candidates(query, top_k)

        metrics_list: List[dict[str, float]] = []
        for source, rec_id, item in raw_results:
            metrics: dict[str, float] = {}
            if source == "error" and rec_id is not None:
                metrics["frequency"] = self._error_frequency(int(rec_id))
            if source == "enhancement":
                metrics["roi"] = self._enhancement_roi(item)
            if source == "workflow":
                metrics["usage"] = self._workflow_usage(item)
            if source == "bot" and rec_id is not None:
                metrics["deploy"] = self._bot_deploy_freq(int(rec_id))
            metrics_list.append(metrics)

        max_vals: dict[str, float] = {}
        for m in metrics_list:
            for k, v in m.items():
                if v > max_vals.get(k, 0.0):
                    max_vals[k] = v

        scored: List[dict[str, Any]] = []
        for (source, rec_id, item), m in zip(raw_results, metrics_list):
            comps: List[float] = []
            for k, v in m.items():
                max_v = max_vals.get(k, 0.0)
                comps.append(v / max_v if max_v else 0.0)
            dist = item["_distance"] if isinstance(item, dict) else getattr(item, "_distance", 0.0)
            # ``search_by_vector`` returns a distance where smaller means more similar.
            # Convert it into a normalised similarity component before combining with
            # any metric based signals gathered above.
            comps.append(1.0 / (1.0 + float(dist)))
            combined_score = sum(comps) / len(comps) if comps else 0.0
            scored.append({
                "source": source,
                "record_id": rec_id,
                "item": item,
                # keep key name ``confidence`` for backwards compatibility
                "confidence": combined_score,
                **m,
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

        hits: List[RetrievalHit] = []
        for idx, entry in enumerate(scored):
            item = entry["item"]
            meta = item if isinstance(item, dict) else item.__dict__
            metrics = {
                k: v
                for k, v in entry.items()
                if k not in {"source", "record_id", "item", "confidence"}
            }
            top_metric = max(metrics, key=metrics.get, default=None)
            reason = reason_map.get(top_metric, "relevant match")
            if idx in link_paths:
                reason = f"{reason} (linked via {link_paths[idx]})"
            hits.append(
                RetrievalHit(
                    source_db=entry["source"],
                    record_id=entry["record_id"],
                    metadata=meta,
                    confidence_score=entry["confidence"],
                    reason=reason,
                )
            )

        hits.sort(key=lambda h: h.confidence_score, reverse=True)
        return hits[:top_k]

    # Backwards compatibility for older callers
    def retrieve_with_confidence(
        self, query: Any, top_k: int = 10, link_multiplier: float = 1.1
    ) -> List[dict[str, Any]]:
        hits = self.retrieve(query, top_k=top_k, link_multiplier=link_multiplier)
        return [
            {
                "source": h.source_db,
                "record_id": h.record_id,
                "item": h.metadata,
                "confidence": h.confidence_score,
                "reason": h.reason,
            }
            for h in hits
        ]
