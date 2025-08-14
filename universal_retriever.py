from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable, List, Sequence


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
    def retrieve(self, query: Any, top_k: int = 10) -> List[Any]:
        """Return top results from all databases for ``query``."""

        vector = self._vector_for_query(query)
        candidates: List[Any] = []
        for db in (
            self.bot_db,
            self.workflow_db,
            self.error_db,
            self.enhancement_db,
            self.information_db,
        ):
            if db is None:
                continue
            try:
                matches = db.search_by_vector(vector, top_k)
            except Exception:  # pragma: no cover - defensive
                continue
            candidates.extend(matches)

        def _dist(item: Any) -> float:
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

    def retrieve_with_confidence(self, query: Any, top_k: int = 10) -> List[dict[str, Any]]:
        """Retrieve results with a combined confidence score."""

        results = self.retrieve(query, top_k)
        metrics_list: List[dict[str, float]] = []
        for item in results:
            metrics: dict[str, float] = {}
            if isinstance(item, dict) and "id" in item and self.error_db:
                metrics["frequency"] = self._error_frequency(int(item["id"]))
            if self.enhancement_db and hasattr(item, "after_code"):
                metrics["roi"] = self._enhancement_roi(item)
            if self.workflow_db and hasattr(item, "performance_data"):
                metrics["usage"] = self._workflow_usage(item)
            if isinstance(item, dict) and "id" in item and self.bot_db and "purpose" in item:
                metrics["deploy"] = self._bot_deploy_freq(int(item["id"]))
            metrics_list.append(metrics)

        max_vals: dict[str, float] = {}
        for m in metrics_list:
            for k, v in m.items():
                if v > max_vals.get(k, 0.0):
                    max_vals[k] = v

        scored: List[dict[str, Any]] = []
        for item, m in zip(results, metrics_list):
            comps: List[float] = []
            for k, v in m.items():
                max_v = max_vals.get(k, 0.0)
                comps.append(v / max_v if max_v else 0.0)
            dist = item["_distance"] if isinstance(item, dict) else getattr(item, "_distance", 0.0)
            comps.append(1.0 / (1.0 + float(dist)))
            confidence = sum(comps) / len(comps) if comps else 0.0
            scored.append({"item": item, "confidence": confidence, **m})
        return scored
