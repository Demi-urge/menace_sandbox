from __future__ import annotations

import json
import sqlite3
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Union, Dict
import logging
import sys

_ALIASES = (
    "universal_retriever",
    "menace.universal_retriever",
    "menace_sandbox.universal_retriever",
)
for _alias in _ALIASES:
    _existing = sys.modules.get(_alias)
    if _existing is not None and _existing is not sys.modules.get(__name__):
        globals().update(_existing.__dict__)
        sys.modules[__name__] = _existing
        __package__ = _alias.rsplit(".", 1)[0]
        break
else:
    _current = sys.modules[__name__]
    for _alias in _ALIASES:
        sys.modules.setdefault(_alias, _current)
if not __package__:
    __package__ = "menace"

try:  # pragma: no cover - optional dependency
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - fallback when running directly
    import metrics_exporter as _me  # type: ignore

try:  # pragma: no cover - lightweight dependency
    from .data_bot import MetricsDB
except Exception:  # pragma: no cover - fallback when module unavailable
    MetricsDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .vector_metrics_db import VectorMetricsDB
except Exception:  # pragma: no cover - fallback when module unavailable
    VectorMetricsDB = None  # type: ignore

_VEC_METRICS = VectorMetricsDB() if VectorMetricsDB is not None else None

try:  # pragma: no cover - typing only
    from .roi_tracker import ROITracker
except Exception:  # pragma: no cover - fallback when tracker unavailable
    ROITracker = None  # type: ignore

try:
    from .metrics_plugins import fetch_retrieval_stats
except Exception:  # pragma: no cover - fallback when unavailable
    def fetch_retrieval_stats(*_args: Any, **_kwargs: Any) -> Dict[str, float]:
        return {"win_rate": 0.0, "regret_rate": 0.0}

_RETRIEVAL_RANK = _me.Gauge(
    "retrieval_rank",
    "Rank position for retrieval results",
    ["origin_db"],
)
# New cumulative counters/histograms exported via ``metrics_exporter``
_RETRIEVAL_HITS_TOTAL = _me.retrieval_hits_total
_RETRIEVAL_TOKENS_INJECTED_TOTAL = _me.retrieval_tokens_injected_total
_RETRIEVAL_RANK_HISTOGRAM = _me.retrieval_rank_histogram
_RETRIEVAL_SCORE = _me.Gauge(
    "retrieval_similarity_score",
    "Similarity score for retrieval result",
    ["origin_db"],
)

_RETRIEVAL_QUERY_TIME = _me.Gauge(
    "retrieval_query_time",
    "Total time taken for retrieval queries",
    [],
)
_RETRIEVAL_DB_TIME = _me.Gauge(
    "retrieval_db_response_time",
    "Time taken by each database to respond",
    ["origin_db"],
)
_RETRIEVAL_HIT_RATE = _me.Gauge(
    "retrieval_hit_rate",
    "Fraction of retrieved results included in final prompt",
    [],
)

logger = logging.getLogger(__name__)


def log_retrieval_metrics(
    origin_db: str,
    record_id: Any,
    rank: int,
    hit: bool,
    tokens: int,
    similarity: float,
    *,
    session_id: str,
) -> None:
    """Log retrieval statistics to Prometheus and VectorMetricsDB."""

    try:
        _RETRIEVAL_RANK.labels(origin_db=origin_db).set(rank)
        _RETRIEVAL_RANK_HISTOGRAM.labels(rank=rank).inc()
        if hit:
            _RETRIEVAL_HITS_TOTAL.inc()
            _RETRIEVAL_TOKENS_INJECTED_TOTAL.inc(tokens)
        _RETRIEVAL_SCORE.labels(origin_db=origin_db).set(similarity)
    except Exception:
        pass  # best effort metrics

    if _VEC_METRICS is not None:
        try:
            _VEC_METRICS.log_retrieval(
                db=origin_db,
                tokens=tokens,
                wall_time_ms=0.0,
                hit=hit,
                rank=rank,
                contribution=similarity,
                prompt_tokens=tokens if hit else 0,
                session_id=session_id,
                vector_id=str(record_id),
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist retrieval metrics")


def _log_stat_to_db(entry: dict[str, Any]) -> None:
    """Persist retrieval statistics for later analysis."""

    try:
        conn = sqlite3.connect("metrics.db")
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_stats (
                    session_id TEXT,
                    origin_db TEXT,
                    record_id TEXT,
                    rank INTEGER,
                    hit INTEGER,
                    hit_rate REAL,
                    tokens_injected INTEGER,
                    contribution REAL,
                    patch_id TEXT,
                    db_source TEXT,
                    ts TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                INSERT INTO retrieval_stats (
                    session_id, origin_db, record_id, rank, hit,
                    hit_rate, tokens_injected, contribution, patch_id, db_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["session_id"],
                    entry["origin_db"],
                    str(entry.get("record_id")),
                    entry["rank"],
                    int(entry["hit"]),
                    entry.get("hit_rate", 0.0),
                    entry.get("tokens_injected", 0),
                    entry.get("contribution"),
                    entry.get("patch_id", ""),
                    entry.get("db_source", ""),
                ),
            )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to log retrieval stat")


def mark_retrieval_contribution(session_id: str, record_id: Any, contribution: float) -> None:
    """Update contribution score for a retrieval result."""

    try:
        conn = sqlite3.connect("metrics.db")
        with conn:
            conn.execute(
                "UPDATE retrieval_stats SET contribution=? WHERE session_id=? AND record_id=?",
                (contribution, session_id, str(record_id)),
            )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to update retrieval contribution")


@dataclass
class ResultBundle:
    """Container for retrieval results.

    Only lightweight, serialisable information is stored: the originating
    database label, a metadata mapping describing the matched record,
    the final combined score and a human readable ``reason``.  Callers can
    inspect the metadata to learn about raw vector distances and contextual
    metrics that contributed to the score.
    """

    origin_db: str
    metadata: dict[str, Any]
    score: float
    reason: str

    # ------------------------------------------------------------------
    # Backwards compatibility helpers
    @property
    def confidence(self) -> float:  # pragma: no cover - simple alias
        return self.score

    @property
    def confidence_score(self) -> float:  # pragma: no cover - simple alias
        return self.score

    @property
    def record_id(self) -> Any:  # pragma: no cover - compatibility shim
        """Best-effort extraction of an identifier from ``metadata``."""

        if isinstance(self.metadata, dict):
            for key in ("id", "wid", "bid", "record_id", "info_id", "item_id"):
                if key in self.metadata:
                    return self.metadata[key]
        return None

    @property
    def links(self) -> List[Any]:  # pragma: no cover - convenience accessor
        """Return identifiers of linked records if present."""

        if isinstance(self.metadata, dict):
            return list(self.metadata.get("linked_records", []) or [])
        return []

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - simple serialiser
        return {
            "origin_db": self.origin_db,
            "metadata": self.metadata,
            "score": self.score,
            "reason": self.reason,
        }


# Older names retained for compatibility
RetrievedItem = ResultBundle
RetrievalHit = ResultBundle


@dataclass
class RetrievalWeights:
    """Tunable weights for ranking and feedback signals."""

    similarity: float = 0.7
    context: float = 0.3
    win: float = 0.1
    regret: float = 0.1
    stale_cost: float = 0.01


def boost_linked_candidates(
    scored: List[dict[str, Any]],
    *,
    bot_db: Any | None = None,
    error_db: Any | None = None,
    information_db: Any | None = None,
    multiplier: float = 1.1,
) -> dict[int, tuple[str, List[Any]]]:
    """Apply a score boost for items linked via bot relationships.

    ``scored`` is modified in-place. Any candidates that share a bot ID
    through the ``bot_workflow``, ``bot_error``, ``bot_enhancement`` or
    ``information_*`` tables receive the ``multiplier`` on their
    ``confidence`` score.  The function returns a mapping of candidate
    index to a tuple of the linkage path string and the related record
    ids for that candidate.
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
        elif typ == "information" and cid and information_db:
            try:
                rows = information_db.conn.execute(
                    "SELECT bot_id FROM information_bots WHERE info_id=?",
                    (int(cid),),
                ).fetchall()
            except sqlite3.Error:
                rows = []
            bots.update(int(r[0]) for r in rows)
            if bot_db:
                try:
                    w_rows = information_db.conn.execute(
                        "SELECT workflow_id FROM information_workflows WHERE info_id=?",
                        (int(cid),),
                    ).fetchall()
                except sqlite3.Error:
                    w_rows = []
                for wid in (int(r[0]) for r in w_rows):
                    b_rows = bot_db.conn.execute(
                        "SELECT bot_id FROM bot_workflow WHERE workflow_id=?",
                        (wid,),
                    ).fetchall()
                    bots.update(int(r[0]) for r in b_rows)
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

    link_info: dict[int, tuple[str, List[Any]]] = {}
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
                link_ids = [scored[n].get("record_id") for n in members if n != m]
                link_info[m] = (path_str, link_ids)

    return link_info


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
        code_db: Any | None = None,
        knowledge_graph: Any | None = None,
        weights: "RetrievalWeights | None" = None,
    ) -> None:
        self.bot_db = bot_db
        self.workflow_db = workflow_db
        self.error_db = error_db
        self.enhancement_db = enhancement_db
        self.information_db = information_db
        self.code_db = code_db
        self.graph = knowledge_graph
        self.weights = weights or RetrievalWeights()
        # lazy-instantiated DeploymentDB connection for bot deployment stats
        self._deploy_db: Any | None = None

        # registry of embeddable databases keyed by origin label
        self._dbs: dict[str, Any] = {}
        self._id_fields: dict[str, tuple[str, ...]] = {}
        self._encoder: Any | None = None
        self._last_db_times: Dict[str, float] = {}

        # register known database types.  The first registration provides the
        # encoder used for free-form text queries.
        self.register_db("bot", bot_db, ("id", "bid"))
        self.register_db("workflow", workflow_db, ("id", "wid"))
        self.register_db("error", error_db, ("id",))
        self.register_db("enhancement", enhancement_db, ("id",))
        self.register_db("information", information_db, ("id", "info_id", "item_id"))

        if self._encoder is None and code_db is None:
            raise ValueError("At least one database instance is required")

    def register_db(
        self, name: str, db: Any | None, id_fields: Sequence[str]
    ) -> None:
        """Register an embeddable database for inclusion in retrieval."""

        if db is None:
            return
        self._dbs[name] = db
        self._id_fields[name] = tuple(id_fields)
        if self._encoder is None:
            self._encoder = db

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

        for name, db in self._dbs.items():
            # Try direct vectorisation of the record instance
            try:
                vec = db.vector(query)
                if vec is not None:
                    return list(vec)
            except Exception:
                pass
            # Fallback to stored vector via identifier lookup
            rec_id = self._extract_id(query, self._id_fields[name])
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

        candidates: List[tuple[str, Any, Any]] = []
        vector: List[float] | None = None
        timings: Dict[str, float] = {}

        if self._dbs:
            vector = self._to_vector(query)
            for source, db in self._dbs.items():
                start = time.perf_counter()
                try:
                    matches = db.search_by_vector(vector, top_k)
                except Exception:  # pragma: no cover - defensive
                    timings[source] = time.perf_counter() - start
                    continue
                timings[source] = time.perf_counter() - start
                for m in matches:
                    rec_id = self._extract_id(m, self._id_fields[source])
                    candidates.append((source, rec_id, m))

        if self.code_db:
            start = time.perf_counter()
            try:
                if hasattr(self.code_db, "search_by_vector"):
                    if vector is None:
                        vector = self._to_vector(query)
                    matches = self.code_db.search_by_vector(vector, top_k)
                elif isinstance(query, str):
                    matches = self.code_db.search(query)[:top_k]
                    for m in matches:
                        if isinstance(m, dict) and "_distance" not in m:
                            m["_distance"] = 0.0
                else:
                    matches = []
            except Exception:  # pragma: no cover - defensive
                matches = []
            timings["code"] = time.perf_counter() - start
            for m in matches:
                rec_id = self._extract_id(m, ("id", "cid"))
                candidates.append(("code", rec_id, m))

        def _dist(entry: tuple[str, Any, Any]) -> float:
            item = entry[2]
            if isinstance(item, dict):
                return float(item.get("_distance", float("inf")))
            return float(getattr(item, "_distance", float("inf")))

        candidates.sort(key=_dist)
        self._last_db_times = timings
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
        """Return ROI uplift score for an enhancement record.

        Uses ``score`` and ``cost_estimate`` fields when available.  Missing
        values are treated as ``0`` so callers only need to provide the fields
        they track.
        """

        if not self.enhancement_db:
            return 0.0
        try:
            score = float(getattr(enh, "score", 0.0) or 0.0)
            cost = float(getattr(enh, "cost_estimate", 0.0) or 0.0)
            return score - cost
        except Exception:
            return 0.0

    def _workflow_usage(self, wf: Any) -> float:
        """Return approximate usage count for a workflow record."""

        # Prefer explicit usage counters when the WorkflowDB exposes them
        if self.workflow_db:
            wid = self._extract_id(wf, ("id", "wid"))
            if wid is not None:
                try:
                    usage = float(self.workflow_db.usage_rate(int(wid)))
                    if usage:
                        return usage
                except Exception:
                    pass

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
        if self._deploy_db is None:
            try:
                from .deployment_bot import DeploymentDB  # lazy import

                self._deploy_db = DeploymentDB("deployment.db")
            except Exception:
                self._deploy_db = None
        if not self._deploy_db:
            return 0.0
        try:
            cur = self._deploy_db.conn.execute(
                "SELECT COUNT(*) FROM bot_trials WHERE bot_id=?",
                (bot_id,),
            ).fetchone()
            if cur and cur[0] is not None:
                return float(cur[0])
        except sqlite3.Error:
            return 0.0
        return 0.0

    # ------------------------------------------------------------------
    def _related_boost(
        self,
        scored: List[dict[str, Any]],
        *,
        multiplier: float = 1.1,
        cap: float = 2.0,
    ) -> dict[int, tuple[str, List[Any]]]:
        """Boost results that are linked via the knowledge graph or tables.

        Parameters
        ----------
        scored:
            Candidate entries accumulated during retrieval.  Each element is
            a mapping containing ``source`` and ``record_id`` keys alongside a
            ``confidence`` score.
        multiplier:
            Factor applied to confidence scores for related records.
        cap:
            Maximum multiple of the original score after all boosting.  This
            prevents runaway amplification when several relationships exist.

        Returns
        -------
        dict
            Mapping of candidate index to ``(path, links)`` tuples. ``path``
            describes the linkage chain while ``links`` lists the related
            record identifiers.
        """

        if multiplier <= 1.0 or len(scored) <= 1:
            return {}

        base_scores = [entry.get("confidence", 0.0) for entry in scored]

        # When a KnowledgeGraph is supplied we use its edges to discover
        # relationships.  Otherwise we fall back to the legacy link-table
        # boosting.
        if self.graph and getattr(self.graph, "graph", None):
            G = self.graph.graph
            nodes: List[str] = []
            type_map: List[str | None] = []
            node_to_idx: dict[str, int] = {}
            for idx, entry in enumerate(scored):
                node = f"{entry.get('source')}:{entry.get('record_id')}"
                nodes.append(node)
                type_map.append(entry.get("source"))
                node_to_idx[node] = idx

            parent = list(range(len(scored)))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for i, node in enumerate(nodes):
                if node not in G:
                    continue
                for nbr in G.neighbors(node):
                    j = node_to_idx.get(nbr)
                    if j is not None:
                        union(i, j)

            groups: dict[int, List[int]] = defaultdict(list)
            for idx in range(len(scored)):
                groups[find(idx)].append(idx)

            link_info: dict[int, tuple[str, List[Any]]] = {}
            order = ["bot", "workflow", "enhancement", "error", "information"]
            for members in groups.values():
                if len(members) > 1:
                    types = {type_map[m] for m in members if type_map[m]}
                    path_types = [t for t in order if t in types]
                    path = "->".join(path_types)
                    for m in members:
                        scored[m]["confidence"] = min(
                            scored[m]["confidence"] * multiplier,
                            base_scores[m] * cap,
                        )
                        link_ids = [
                            scored[n].get("record_id") for n in members if n != m
                        ]
                        link_info[m] = (path, link_ids)
            return link_info

        link_info = boost_linked_candidates(
            scored,
            bot_db=self.bot_db,
            error_db=self.error_db,
            information_db=self.information_db,
            multiplier=multiplier,
        )
        for idx, entry in enumerate(scored):
            entry["confidence"] = min(entry["confidence"], base_scores[idx] * cap)
        return link_info

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
            # Large usage numbers can otherwise dominate the combined score
            # and overshadow direct similarity.  We therefore dampen the
            # contribution by blending with a constant denominator so that
            # extremely heavy workflows do not automatically eclipse the
            # originating bot.
            score = lu / (5.0 + lu)
        elif kind == "bot":
            bid = self._extract_id(record, ("id", "bid"))
            if bid is not None:
                freq = self._bot_deploy_freq(int(bid))
                metrics["deploy"] = freq
                lf = math.log1p(freq)
                score = lf / (1.0 + lf)
        elif kind == "code":
            comp = float(
                getattr(record, "complexity", getattr(record, "complexity_score", 0.0))
                or 0.0
            )
            metrics["complexity"] = comp
            lc = math.log1p(max(comp, 0.0))
            score = lc / (1.0 + lc)
        return score, metrics

    def retrieve(
        self,
        query: Any,
        top_k: int = 10,
        link_multiplier: float = 1.1,
        return_metrics: bool = False,
        roi_tracker: "ROITracker | None" = None,
        adjust_weights: bool = False,
    ) -> Union[List[ResultBundle], Tuple[List[ResultBundle], List[dict[str, Any]]]]:
        """Retrieve results with scores and reasons.

        Beyond returning :class:`ResultBundle` objects, this method records
        rich statistics for every candidate.  Each hit stores its rank
        position, whether it was injected into downstream prompts, the number
        of tokens contributed, and the overall retrieval hit rate.  These
        details are persisted to the ``retrieval_stats`` SQLite table and
        surfaced via Prometheus metrics such as
        ``retrieval_hits_total`` and ``retrieval_rank_histogram`` so that
        later systems can evaluate contribution scores and win rates.
        """

        start_time = time.perf_counter()
        session_id = uuid.uuid4().hex
        raw_results = self._retrieve_candidates(query, top_k)
        db_times = getattr(self, "_last_db_times", {})
        bias_map: Dict[str, float] = {}
        if roi_tracker is not None:
            try:
                bias_map = roi_tracker.retrieval_bias()
            except Exception:
                bias_map = {}
        feedback_map: Dict[str, Dict[str, float]] = {}
        if MetricsDB is not None:
            try:
                feedback_map = MetricsDB().latest_retriever_kpi()
            except Exception:
                feedback_map = {}

        SIM_WEIGHT = self.weights.similarity
        CTX_WEIGHT = self.weights.context
        WIN_WEIGHT = self.weights.win
        REGRET_WEIGHT = self.weights.regret
        STALE_COST = self.weights.stale_cost

        if adjust_weights:
            try:
                stats = fetch_retrieval_stats()
                if stats.get("count", 0.0) > 0:
                    WIN_WEIGHT *= 1.0 + float(stats.get("win_rate", 0.0))
                    REGRET_WEIGHT *= 1.0 + float(stats.get("regret_rate", 0.0))
            except Exception:
                logger.exception("failed to adjust retrieval weights")

        scored: List[dict[str, Any]] = []
        for source, rec_id, item in raw_results:
            dist = item["_distance"] if isinstance(item, dict) else getattr(item, "_distance", 0.0)
            similarity = 1.0 / (1.0 + float(dist))
            ctx_score, metrics = self._context_score(source, item)
            kpi = feedback_map.get(source, {})
            win_rate = kpi.get("win_rate", 0.0)
            regret_rate = kpi.get("regret_rate", 0.0)
            stale_cost = kpi.get("stale_cost", kpi.get("stale_penalty", 0.0))
            feedback_bias = 1.0 + WIN_WEIGHT * win_rate - REGRET_WEIGHT * regret_rate
            feedback_bias *= math.exp(-STALE_COST * stale_cost)
            combined_score = similarity * SIM_WEIGHT + ctx_score * CTX_WEIGHT
            combined_score *= bias_map.get(source, 1.0) * feedback_bias
            scored.append({
                "source": source,
                "record_id": rec_id,
                "item": item,
                "confidence": combined_score,
                "distance": dist,
                "similarity": similarity,
                "context": ctx_score,
                "win_rate": win_rate,
                "regret_rate": regret_rate,
                "stale_cost": stale_cost,
                **metrics,
            })

        link_info = self._related_boost(
            scored,
            multiplier=link_multiplier,
        )

        reason_map = {
            "similarity": "high vector similarity",
            "roi": "high ROI uplift",
            "frequency": "frequent error recurrence",
            "usage": "heavy usage",
            "deploy": "widely deployed bot",
            "complexity": "high code complexity",
            "win_rate": "high historical win rate",
            "regret_rate": "low regret rate",
            "stale_cost": "fresh embedding",
        }

        hits: List[ResultBundle] = []
        for idx, entry in enumerate(scored):
            item = entry["item"]
            base_meta = item if isinstance(item, dict) else item.__dict__
            meta = dict(base_meta)
            metrics = {
                k: v
                for k, v in entry.items()
                if k
                not in {
                    "source",
                    "record_id",
                    "item",
                    "confidence",
                    "distance",
                    "similarity",
                    "context",
                }
            }
            meta.update(
                {
                    "vector_distance": entry.get("distance", 0.0),
                    "similarity": entry.get("similarity", 0.0),
                    "context_score": entry.get("context", 0.0),
                    "contextual_metrics": metrics,
                }
            )
            metrics_for_reason = {**metrics, "similarity": entry.get("similarity", 0.0)}
            top_metric = max(
                metrics_for_reason, key=metrics_for_reason.get, default=None
            )
            reason = reason_map.get(top_metric, "relevant match")
            if top_metric:
                reason += f" ({top_metric}={metrics_for_reason[top_metric]:.2f})"
            if idx in link_info:
                path, links = link_info[idx]
                reason += f" linked via {path}"
                meta["linked_records"] = links
            hits.append(
                ResultBundle(
                    origin_db=entry["source"],
                    metadata=meta,
                    score=entry["confidence"],
                    reason=reason,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)

        metrics_list: List[dict[str, Any]] = []
        dataset_entries: List[dict[str, Any]] = []
        for rank, bundle in enumerate(hits, start=1):
            tokens = len(json.dumps(bundle.metadata, ensure_ascii=False)) // 4
            included = rank <= top_k
            similarity = float(bundle.metadata.get("similarity", 0.0))
            win_rate = float(bundle.metadata.get("win_rate", 0.0))
            regret_rate = float(bundle.metadata.get("regret_rate", 0.0))
            log_retrieval_metrics(
                bundle.origin_db,
                bundle.record_id,
                rank,
                included,
                tokens,
                similarity,
                session_id=session_id,
            )
            logger.info(
                "retrieval result rank=%d db=%s tokens=%d",
                rank,
                bundle.origin_db,
                tokens,
            )
            metrics_entry = {
                "origin_db": bundle.origin_db,
                "record_id": bundle.record_id,
                "rank": rank,
                "rank_position": rank,
                "hit": included,
                "tokens": tokens,
                "tokens_injected": tokens if included else 0,
                "similarity": similarity,
                "win_rate": win_rate,
                "regret_rate": regret_rate,
                "prompt_tokens": tokens if included else 0,
                "session_id": session_id,
            }
            metrics_list.append(metrics_entry)
            dataset_entries.append(
                {
                    "origin_db": bundle.origin_db,
                    "record_id": bundle.record_id,
                    "rank": rank,
                    "score": bundle.score,
                    "win_rate": win_rate,
                    "regret_rate": regret_rate,
                    "hit": included,
                    "session_id": session_id,
                }
            )

        results = hits[:top_k]

        total_candidates = len(hits)
        hit_rate = len(results) / total_candidates if total_candidates else 0.0
        try:
            _RETRIEVAL_HIT_RATE.set(hit_rate)
        except Exception:
            pass
        if MetricsDB is not None:
            try:
                MetricsDB().log_eval("retrieval_call", "hit_rate", hit_rate)
            except Exception:
                logger.exception("failed to persist hit rate")

        for entry in metrics_list:
            entry["hit_rate"] = hit_rate
            entry["contribution"] = 0.0
            _log_stat_to_db(entry)
        for entry in dataset_entries:
            entry["hit_rate"] = hit_rate

        dataset_path = Path(__file__).resolve().parent / "analytics" / "retrieval_outcomes.jsonl"
        try:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_path.open("a", encoding="utf8") as fh:
                for rec in dataset_entries:
                    fh.write(json.dumps(rec) + "\n")
        except Exception:
            logger.exception("failed to store retrieval outcomes dataset")

        for db_name, duration in db_times.items():
            try:
                _RETRIEVAL_DB_TIME.labels(origin_db=db_name).set(duration)
            except Exception:
                pass
            if MetricsDB is not None:
                try:
                    MetricsDB().log_eval(db_name, "response_time", duration)
                except Exception:
                    logger.exception("failed to persist db timing")
            logger.info("retrieval db=%s response_time=%.6f", db_name, duration)

        total_time = time.perf_counter() - start_time
        try:
            _RETRIEVAL_QUERY_TIME.set(total_time)
        except Exception:
            pass
        if MetricsDB is not None:
            try:
                MetricsDB().log_eval("retrieval_call", "query_time", total_time)
            except Exception:
                logger.exception("failed to persist query time")
        logger.info("retrieval total_query_time=%.6f", total_time)

        if return_metrics:
            return results, metrics_list
        return results

    # Backwards compatibility for older callers
    def retrieve_with_confidence(
        self,
        query: Any,
        top_k: int = 10,
        link_multiplier: float = 1.1,
        return_metrics: bool = False,
        roi_tracker: "ROITracker | None" = None,
        adjust_weights: bool = False,
    ) -> Union[List[dict[str, Any]], Tuple[List[dict[str, Any]], List[dict[str, Any]]]]:
        res = self.retrieve(
            query,
            top_k=top_k,
            link_multiplier=link_multiplier,
            return_metrics=return_metrics,
            roi_tracker=roi_tracker,
            adjust_weights=adjust_weights,
        )
        if return_metrics:
            hits, metrics_list = res  # type: ignore[misc]
        else:
            hits = res  # type: ignore[assignment]
            metrics_list = []
        formatted = [
            {
                "source": h.origin_db,
                "record_id": h.record_id,
                "item": h.metadata,
                "confidence": h.score,
                "reason": h.reason,
            }
            for h in hits
        ]
        if return_metrics:
            return formatted, metrics_list
        return formatted
