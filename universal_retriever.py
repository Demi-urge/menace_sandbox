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
import os
from datetime import datetime
from governed_retrieval import govern_retrieval
import joblib
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from scope_utils import build_scope_clause
from dynamic_path_router import get_project_roots, resolve_path

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

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback when event bus unavailable
    UnifiedEventBus = None  # type: ignore

MENACE_ID = "universal_retriever"

def _default_local_db_path(menace_id: str) -> str:
    """Return a writable fallback path for the local sqlite database.

    The historical behaviour attempted to resolve the sqlite file at import
    time via :func:`dynamic_path_router.resolve_path`.  On fresh checkouts where
    the database had not yet been created this raised ``FileNotFoundError`` and
    prevented ``universal_retriever`` from importing at all.  We still prefer a
    repository-managed location when present, but fall back to creating an
    empty sqlite file in a writable directory so importers can proceed.
    """

    db_name = f"menace_{menace_id}_local.db"

    try:
        return str(resolve_path(db_name))
    except FileNotFoundError:
        pass

    module_dir = Path(__file__).resolve().parent
    module_path = module_dir / db_name

    if module_path.exists():
        return str(module_path)

    try:
        module_dir.mkdir(parents=True, exist_ok=True)
        module_path.touch(exist_ok=True)
        return str(module_path)
    except OSError:
        for root in get_project_roots():
            candidate = root / db_name
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.touch(exist_ok=True)
                return str(candidate)
            except OSError:
                continue
        raise


LOCAL_DB_PATH = os.getenv("MENACE_LOCAL_DB_PATH", _default_local_db_path(MENACE_ID))
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
router: DBRouter = GLOBAL_ROUTER or init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)
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
    context_score: float,
    age: float,
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
                contribution=0.0,
                prompt_tokens=tokens if hit else 0,
                session_id=session_id,
                vector_id=str(record_id),
                similarity=similarity,
                context_score=context_score,
                age=age,
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist retrieval metrics")


def _log_stat_to_db(entry: dict[str, Any]) -> None:
    """Persist retrieval statistics for later analysis."""

    try:
        conn = router.get_connection("retrieval_stats")
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_stats (
                    session_id TEXT,
                    origin_db TEXT,
                    record_id TEXT,
                    vector_id TEXT,
                    db_type TEXT,
                    rank INTEGER,
                    hit INTEGER,
                    hit_rate REAL,
                    tokens_injected INTEGER,
                    contribution REAL,
                    patch_id TEXT,
                    db_source TEXT,
                    age REAL,
                    similarity REAL,
                    frequency REAL,
                    roi_delta REAL,
                    usage REAL,
                    prior_hits INTEGER,
                    ts TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """,
            )
            cols = [r[1] for r in conn.execute("PRAGMA table_info(retrieval_stats)").fetchall()]
            migrations = {
                "vector_id": "ALTER TABLE retrieval_stats ADD COLUMN vector_id TEXT",
                "db_type": "ALTER TABLE retrieval_stats ADD COLUMN db_type TEXT",
                "age": "ALTER TABLE retrieval_stats ADD COLUMN age REAL",
                "similarity": "ALTER TABLE retrieval_stats ADD COLUMN similarity REAL",
                "frequency": "ALTER TABLE retrieval_stats ADD COLUMN frequency REAL",
                "roi_delta": "ALTER TABLE retrieval_stats ADD COLUMN roi_delta REAL",
                "usage": "ALTER TABLE retrieval_stats ADD COLUMN usage REAL",
                "prior_hits": "ALTER TABLE retrieval_stats ADD COLUMN prior_hits INTEGER",
            }
            for col, stmt in migrations.items():
                if col not in cols:
                    conn.execute(stmt)
            conn.execute(
                """
                INSERT INTO retrieval_stats (
                    session_id, origin_db, record_id, vector_id, db_type,
                    rank, hit, hit_rate, tokens_injected, contribution,
                    patch_id, db_source, age, similarity, frequency,
                    roi_delta, usage, prior_hits
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["session_id"],
                    entry["origin_db"],
                    str(entry.get("record_id")),
                    str(entry.get("vector_id", "")),
                    entry.get("db_type", ""),
                    entry["rank"],
                    int(entry["hit"]),
                    entry.get("hit_rate", 0.0),
                    entry.get("tokens_injected", 0),
                    entry.get("contribution"),
                    entry.get("patch_id", ""),
                    entry.get("db_source", ""),
                    entry.get("age"),
                    entry.get("similarity"),
                    entry.get("frequency"),
                    entry.get("roi_delta"),
                    entry.get("usage"),
                    entry.get("prior_hits", 0),
                ),
            )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to log retrieval stat")


def mark_retrieval_contribution(session_id: str, record_id: Any, contribution: float) -> None:
    """Update contribution score for a retrieval result."""

    try:
        conn = router.get_connection("retrieval_stats")
        with conn:
            conn.execute(
                "UPDATE retrieval_stats SET contribution=? WHERE session_id=? AND record_id=?",
                (contribution, session_id, str(record_id)),
            )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to update retrieval contribution")


def _prior_hit_count(origin_db: str, record_id: Any) -> int:
    """Return how many times a vector has previously been a hit."""

    try:
        conn = router.get_connection("retrieval_stats")
        cur = conn.execute(
            "SELECT COUNT(*) FROM retrieval_stats WHERE origin_db=? AND record_id=? AND hit=1",
            (origin_db, str(record_id)),
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:  # pragma: no cover - best effort
        return 0


def _win_regret_rates(origin_db: str, record_id: Any) -> Tuple[float, float]:
    """Return historical win/regret rates for a vector."""

    if _VEC_METRICS is None:
        return 0.0, 0.0
    try:
        cur = _VEC_METRICS.conn.execute(
            """
            SELECT AVG(win), AVG(regret)
              FROM vector_metrics
             WHERE event_type='retrieval' AND db=? AND vector_id=?
            """,
            (origin_db, str(record_id)),
        )
        row = cur.fetchone()
        if row:
            w, r = row
            return float(w or 0.0), float(r or 0.0)
    except Exception:  # pragma: no cover - best effort
        return 0.0, 0.0
    return 0.0, 0.0


def load_ranker(path: str | Path) -> Any:
    """Load a serialized ranking model from ``path`` stored via joblib."""

    return joblib.load(path)


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


class RetrievalResult(list):
    """List of retrieval hits carrying session metadata."""

    def __init__(
        self,
        items: Sequence[Any],
        session_id: str,
        vectors: List[Tuple[str, str, float]],
        fallback_sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(items)
        self.session_id = session_id
        self.vectors = vectors
        self.fallback_sources = list(fallback_sources or [])


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
        model_path: str | Path | None = None,
        ranker: Any | None = None,
        reliability_threshold: float = 0.0,
        fallback_on_low_reliability: bool = True,
        enable_model_ranking: bool = True,
        enable_reliability_bias: bool = True,
        event_bus: UnifiedEventBus | None = None,
    ) -> None:
        self.bot_db = bot_db
        self.workflow_db = workflow_db
        self.error_db = error_db
        self.enhancement_db = enhancement_db
        self.information_db = information_db
        self.code_db = code_db
        self.graph = knowledge_graph
        self.weights = weights or RetrievalWeights()
        self.reliability_threshold = float(reliability_threshold)
        self.fallback_on_low_reliability = bool(fallback_on_low_reliability)
        self._ranker_model: Any | None = None
        self.use_ranker = bool(enable_model_ranking)
        if enable_model_ranking:
            if ranker is not None:
                self._ranker_model = ranker
            elif model_path:
                try:
                    self._ranker_model = load_ranker(model_path)
                except Exception:
                    logger.exception("failed to load ranking model from %s", model_path)
        self.enable_reliability_bias = bool(enable_reliability_bias)
        self._reliability_stats: Dict[str, Dict[str, float]] = {}
        # lazy-instantiated DeploymentDB connection for bot deployment stats
        self._deploy_db: Any | None = None

        # registry of embeddable databases keyed by origin label
        self._dbs: dict[str, Any] = {}
        self._id_fields: dict[str, tuple[str, ...]] = {}
        self._encoder: Any | None = None
        self._last_db_times: Dict[str, float] = {}
        self._last_fallback_sources: List[str] = []

        # register known database types.  The first registration provides the
        # encoder used for free-form text queries.
        self.register_db("bot", bot_db, ("id", "bid"))
        self.register_db("workflow", workflow_db, ("id", "wid"))
        self.register_db("error", error_db, ("id",))
        self.register_db("enhancement", enhancement_db, ("id",))
        self.register_db("information", information_db, ("id", "info_id", "item_id"))

        self._load_reliability_stats()
        if self.enable_reliability_bias and self._reliability_stats:
            self._dbs = dict(
                sorted(
                    self._dbs.items(),
                    key=lambda kv: self._reliability_stats.get(kv[0], {})
                    .get("reliability", 0.0),
                    reverse=True,
                )
            )

        if self._encoder is None and code_db is None:
            raise ValueError("At least one database instance is required")

        self.event_bus = event_bus
        if self.event_bus is None and UnifiedEventBus is not None:
            try:
                self.event_bus = UnifiedEventBus()
            except Exception:
                self.event_bus = None
        if self.event_bus is not None:
            try:
                self.event_bus.subscribe(
                    "retrieval:feedback", lambda *_: self.reload_reliability_scores()
                )
            except Exception:
                pass

    @property
    def reliability_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return reliability statistics for all registered databases.

        The metrics are loaded from :class:`MetricsDB` on demand so that
        callers can always access up‑to‑date win/regret rates even when
        ``retrieve`` has not been invoked yet.
        """

        if not self._reliability_stats:
            # ``_load_reliability_stats`` swallows any database errors and
            # simply leaves ``_reliability_stats`` empty on failure.  This
            # makes the property safe to call in best‑effort contexts.
            self._load_reliability_stats()
        return dict(self._reliability_stats)

    def reload_ranker_model(self, model_path: str | Path) -> None:
        """Reload the ranking model from ``model_path`` saved via joblib.

        Any errors are logged but otherwise ignored to keep the retriever
        operational.
        """
        try:
            self._ranker_model = load_ranker(model_path)
        except Exception:
            logger.exception("failed to reload ranking model from %s", model_path)

    def reload_reliability_scores(self) -> None:
        """Refresh reliability metrics used for ranking.

        Both the win/regret rates stored in :class:`MetricsDB` and the
        lightweight counters in :class:`vector_metrics_db.VectorMetricsDB`
        may change over time as new retrieval results are observed.  This
        helper reloads the cached values from :class:`MetricsDB` and, when
        available, touches the vector metrics backend so external schedulers
        can notify the retriever that updated statistics should be used.
        """

        # Refresh the cached MetricsDB values so callers can observe the
        # latest reliability information without needing to trigger a full
        # retrieval cycle.
        self._load_reliability_stats()

        if _VEC_METRICS is None:
            return
        for name in list(self._dbs):
            try:
                _VEC_METRICS.retriever_win_rate(name)
                _VEC_METRICS.retriever_regret_rate(name)
            except Exception:
                continue

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
    def _model_predict(self, source: str, feats: Dict[str, float]) -> float:
        """Return ranking model probability for candidate ``source``."""
        model = self._ranker_model
        if not model or not feats:
            return 1.0
        try:
            aliases = {
                "embedding_age": "age",
                "vector_similarity": "similarity",
                "workflow_frequency": "exec_freq",
                "prior_hit_count": "prior_hits",
            }
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                feature_names = getattr(model, "get", lambda *_: None)("features", None)
            vec: List[float] = []
            for name in feature_names or []:
                if name.startswith("db_"):
                    vec.append(1.0 if source == name[3:] else 0.0)
                else:
                    key = aliases.get(name, name)
                    vec.append(float(feats.get(key, 0.0)))
            if hasattr(model, "coef_"):
                coef = getattr(model, "coef_", [[0.0]])[0]
                intercept = getattr(model, "intercept_", [0.0])[0]
                z = sum(c * v for c, v in zip(coef, vec)) + intercept
                return 1.0 / (1.0 + math.exp(-z))
            if hasattr(model, "predict_proba"):
                import numpy as np

                proba = model.predict_proba(np.array([vec]))[0]
                return float(proba[1] if len(proba) > 1 else proba[0])
            if isinstance(model, dict):  # backward compatibility
                z = sum(c * v for c, v in zip(model.get("coef", []), vec)) + model.get(
                    "intercept", 0.0
                )
                return 1.0 / (1.0 + math.exp(-z))
        except Exception:
            logger.exception("ranking model prediction failed")
        return 1.0

    def _load_reliability_stats(self) -> Dict[str, Dict[str, float]]:
        """Fetch win/regret rates for each DB from ``MetricsDB``.

        Results are cached on ``self._reliability_stats`` and returned as a
        mapping of database name to ``{"win_rate", "regret_rate", "reliability"}``.
        Any database access errors are swallowed and result in an empty
        mapping.
        """

        stats: Dict[str, Dict[str, float]] = {}
        if _VEC_METRICS is not None:
            try:
                win_map = _VEC_METRICS.retriever_win_rate_by_db()
                regret_map = _VEC_METRICS.retriever_regret_rate_by_db()
                for name in set(win_map) | set(regret_map):
                    win_rate = float(win_map.get(name, 0.0))
                    regret_rate = float(regret_map.get(name, 0.0))
                    stats[name] = {
                        "win_rate": win_rate,
                        "regret_rate": regret_rate,
                        "reliability": win_rate - regret_rate,
                        "sample_count": 0.0,
                    }
            except Exception:
                stats = {}
        if not stats and MetricsDB is not None:
            try:
                MetricsDB()
                with router.get_connection("retriever_stats") as conn:
                    cur = conn.execute(
                        "SELECT origin_db, wins, regrets FROM retriever_stats"
                    )
                    for origin, wins, regrets in cur.fetchall():
                        total = float(wins) + float(regrets)
                        win_rate = float(wins) / total if total else 0.0
                        regret_rate = float(regrets) / total if total else 0.0
                        stats[str(origin)] = {
                            "win_rate": win_rate,
                            "regret_rate": regret_rate,
                            "reliability": win_rate - regret_rate,
                            "sample_count": total,
                        }
            except Exception:
                stats = {}
        self._reliability_stats = stats
        return self._reliability_stats

    # ------------------------------------------------------------------
    def _retrieve_candidates(
        self, query: Any, top_k: int = 10, db_names: Sequence[str] | None = None
    ) -> List[tuple[str, Any, Any, float, Dict[str, float]]]:
        """Return candidate scores and feature maps for ``query``.

        Each candidate is described by a feature vector mirroring the
        statistics used during model training.  The ranking model can later
        score the candidate using these features.
        """

        candidates: List[tuple[str, Any, Any, float, Dict[str, float]]] = []
        vector: List[float] | None = None
        timings: Dict[str, float] = {}
        fallback_sources: List[str] = []
        reliability_map = self._reliability_stats if self.enable_reliability_bias else {}
        kpi_map: Dict[str, Dict[str, float]] = {}
        if MetricsDB is not None:
            try:
                kpi_map = MetricsDB().latest_retriever_kpi()
            except Exception:
                kpi_map = {}

        SIM_WEIGHT = self.weights.similarity
        CTX_WEIGHT = self.weights.context

        if self._dbs:
            vector = self._to_vector(query)
            if db_names is not None:
                items = [(n, self._dbs[n]) for n in db_names if n in self._dbs]
            else:
                items = list(self._dbs.items())
            if self.enable_reliability_bias and self._reliability_stats:
                items.sort(
                    key=lambda kv: self._reliability_stats.get(kv[0], {}).get(
                        "reliability", 0.0
                    ),
                    reverse=True,
                )
                if self.reliability_threshold and not self.fallback_on_low_reliability:
                    items = [
                        kv
                        for kv in items
                        if self._reliability_stats.get(kv[0], {}).get(
                            "reliability", 0.0
                        )
                        >= self.reliability_threshold
                    ]
            elif _VEC_METRICS is not None:
                reliabilities: Dict[str, float] = {}
                for name, _ in items:
                    try:
                        win = _VEC_METRICS.retriever_win_rate(name)
                        regret = _VEC_METRICS.retriever_regret_rate(name)
                        reliabilities[name] = win - regret
                    except Exception:
                        reliabilities[name] = 0.0
                items.sort(
                    key=lambda kv: reliabilities.get(kv[0], 0.0), reverse=True
                )
                if self.reliability_threshold and not self.fallback_on_low_reliability:
                    items = [
                        kv
                        for kv in items
                        if reliabilities.get(kv[0], 0.0) >= self.reliability_threshold
                    ]
            fallback_source: str | None = None
            if self.reliability_threshold and self._reliability_stats and items:
                top_name = items[0][0]
                top_rel = self._reliability_stats.get(top_name, {}).get(
                    "reliability", 0.0
                )
                if top_rel < self.reliability_threshold:
                    for name, _ in items[1:]:
                        if (
                            self._reliability_stats.get(name, {}).get(
                                "reliability", 0.0
                            )
                            >= self.reliability_threshold
                        ):
                            fallback_source = name
                            break
            processed: set[str] = set()
            for source, db in items:
                start = time.perf_counter()
                try:
                    matches = db.search_by_vector(vector, top_k)
                except Exception:  # pragma: no cover - defensive
                    timings[source] = time.perf_counter() - start
                    processed.add(source)
                    if (
                        len(candidates) >= top_k
                        and (not fallback_source or fallback_source in processed)
                    ):
                        break
                    continue
                timings[source] = time.perf_counter() - start
                for m in matches:
                    rec_id = self._extract_id(m, self._id_fields[source])
                    dist = (
                        m.get("_distance", 0.0)
                        if isinstance(m, dict)
                        else getattr(m, "_distance", 0.0)
                    )
                    similarity = 1.0 / (1.0 + float(dist))
                    ctx_score, extra = self._context_score(source, m)
                    rel = reliability_map.get(source, {})
                    win_rate = rel.get("win_rate", 0.0)
                    regret_rate = rel.get("regret_rate", 0.0)
                    kpi = kpi_map.get(source, {})
                    stale_cost = kpi.get("stale_cost", kpi.get("stale_penalty", 0.0))
                    samples = rel.get("sample_count", kpi.get("sample_count", 0.0))
                    created_at = (
                        m.get("created_at")
                        if isinstance(m, dict)
                        else getattr(m, "created_at", None)
                    ) or (
                        m.get("ts")
                        if isinstance(m, dict)
                        else getattr(m, "ts", None)
                    ) or (
                        m.get("timestamp")
                        if isinstance(m, dict)
                        else getattr(m, "timestamp", None)
                    )
                    age = 0.0
                    if created_at:
                        try:
                            age = (
                                datetime.utcnow() - datetime.fromisoformat(str(created_at))
                            ).total_seconds()
                        except Exception:
                            age = 0.0
                    exec_freq = float(
                        extra.get("frequency")
                        or extra.get("usage")
                        or extra.get("deploy")
                        or extra.get("exec_freq")
                        or 0.0
                    )
                    roi_delta = float(extra.get("roi") or extra.get("roi_delta") or 0.0)
                    prior_hits = float(_prior_hit_count(source, rec_id))
                    win_hist, regret_hist = _win_regret_rates(source, rec_id)
                    try:
                        severity = float(
                            extra.get("alignment_severity")
                            or (
                                m.get("alignment_severity")
                                if isinstance(m, dict)
                                else getattr(m, "alignment_severity", 0.0)
                            )
                            or 0.0
                        )
                    except Exception:
                        severity = 0.0
                    feats = {
                        "similarity": similarity,
                        "context_score": ctx_score,
                        "win_rate": win_rate,
                        "regret_rate": regret_rate,
                        "stale_cost": stale_cost,
                        "sample_count": samples,
                        "age": age,
                        "exec_freq": exec_freq,
                        "roi_delta": roi_delta,
                        "prior_hits": prior_hits,
                        "alignment_severity": severity,
                        "win": win_hist,
                        "regret": regret_hist,
                        **{
                            k: v
                            for k, v in extra.items()
                            if k not in {"alignment_severity", "win", "regret"}
                        },
                        "distance": dist,
                    }
                    base_score = similarity * SIM_WEIGHT + ctx_score * CTX_WEIGHT
                    candidates.append((source, rec_id, m, base_score, feats))
                processed.add(source)
                if len(candidates) >= top_k and (
                    not fallback_source or fallback_source in processed
                ):
                    break
            if fallback_source and fallback_source in processed:
                fallback_sources.append(fallback_source)

        if self.code_db and len(candidates) < top_k:
            allow = True
            if _VEC_METRICS is not None:
                try:
                    win = _VEC_METRICS.retriever_win_rate("code")
                    regret = _VEC_METRICS.retriever_regret_rate("code")
                    rel = win - regret
                    if (
                        self.reliability_threshold
                        and not self.fallback_on_low_reliability
                        and rel < self.reliability_threshold
                    ):
                        allow = False
                except Exception:
                    pass
            if allow:
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
                    dist = (
                        m.get("_distance", 0.0)
                        if isinstance(m, dict)
                        else getattr(m, "_distance", 0.0)
                    )
                    similarity = 1.0 / (1.0 + float(dist))
                    ctx_score, extra = self._context_score("code", m)
                    rel = reliability_map.get("code", {})
                    win_rate = rel.get("win_rate", 0.0)
                    regret_rate = rel.get("regret_rate", 0.0)
                    kpi = kpi_map.get("code", {})
                    stale_cost = kpi.get("stale_cost", kpi.get("stale_penalty", 0.0))
                    samples = rel.get("sample_count", kpi.get("sample_count", 0.0))
                    created_at = (
                        m.get("created_at")
                        if isinstance(m, dict)
                        else getattr(m, "created_at", None)
                    ) or (
                        m.get("ts")
                        if isinstance(m, dict)
                        else getattr(m, "ts", None)
                    ) or (
                        m.get("timestamp")
                        if isinstance(m, dict)
                        else getattr(m, "timestamp", None)
                    )
                    age = 0.0
                    if created_at:
                        try:
                            age = (
                                datetime.utcnow() - datetime.fromisoformat(str(created_at))
                            ).total_seconds()
                        except Exception:
                            age = 0.0
                    exec_freq = float(
                        extra.get("frequency")
                        or extra.get("usage")
                        or extra.get("deploy")
                        or extra.get("exec_freq")
                        or 0.0
                    )
                    roi_delta = float(extra.get("roi") or extra.get("roi_delta") or 0.0)
                    prior_hits = float(_prior_hit_count("code", rec_id))
                    feats = {
                        "similarity": similarity,
                        "context_score": ctx_score,
                        "win_rate": win_rate,
                        "regret_rate": regret_rate,
                        "stale_cost": stale_cost,
                        "sample_count": samples,
                        "age": age,
                        "exec_freq": exec_freq,
                        "roi_delta": roi_delta,
                        "prior_hits": prior_hits,
                        **extra,
                        "distance": dist,
                    }
                    base_score = similarity * SIM_WEIGHT + ctx_score * CTX_WEIGHT
                    candidates.append(("code", rec_id, m, base_score, feats))
        candidates.sort(key=lambda entry: entry[3], reverse=True)
        self._last_db_times = timings
        self._last_fallback_sources = fallback_sources
        return candidates[:top_k]

    # ------------------------------------------------------------------
    def _error_frequency(self, error_id: int, scope: str | None = None) -> float:
        """Return raw error frequency from ``ErrorDB`` filtered by menace ID."""

        if not self.error_db:
            return 0.0
        try:
            router = getattr(self.error_db, "router", None)
            menace_id = getattr(router, "menace_id", None)
            if menace_id is None:
                config_obj = getattr(self, "config", None)
                menace_id = getattr(config_obj, "menace_id", None)
            if menace_id is None:
                menace_id = os.getenv("MENACE_ID", "")
            clause, params = build_scope_clause("errors", scope or "local", menace_id)
            query = "SELECT frequency FROM errors WHERE id=?"
            if clause:
                query += f" AND {clause}"
            cur = self.error_db.conn.execute(query, (error_id, *params)).fetchone()
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
            except BaseException:
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
        dbs: Sequence[str] | None = None,
    ) -> Union[
        Tuple["RetrievalResult", str, List[Tuple[str, str, float]]],
        Tuple["RetrievalResult", str, List[Tuple[str, str, float]], List[dict[str, Any]]],
    ]:
        """Retrieve results with scores and reasons.

        Returns
        -------
        ``(results, session_id, vectors)`` or
        ``(results, session_id, vectors, metrics)`` when ``return_metrics`` is
        true.  ``vectors`` is a list of ``(db_name, vector_id, score)`` tuples for all
        retrieved items.  ``session_id`` uniquely identifies the retrieval
        session.

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
        if self.enable_reliability_bias:
            self._load_reliability_stats()
        raw_results = self._retrieve_candidates(query, top_k, db_names=dbs)
        db_times = dict(getattr(self, "_last_db_times", {}))
        fb_sources = list(getattr(self, "_last_fallback_sources", []))
        bias_map: Dict[str, float] = {}
        if roi_tracker is not None:
            try:
                bias_map = roi_tracker.retrieval_bias()
            except Exception:
                bias_map = {}
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

        def _score_results(raw: Iterable[tuple[str, Any, Any, float, Dict[str, float]]]) -> None:
            for source, rec_id, item, base_score, feats in raw:
                dist = float(feats.get("distance", 0.0))
                similarity = float(feats.get("similarity", 0.0))
                ctx_score = float(feats.get("context_score", 0.0))
                win_rate = float(feats.get("win_rate", 0.0))
                regret_rate = float(feats.get("regret_rate", 0.0))
                stale_cost = float(feats.get("stale_cost", 0.0))
                sample_count = float(feats.get("sample_count", 0.0))
                metrics = {
                    k: v
                    for k, v in feats.items()
                    if k
                    not in {
                        "distance",
                        "similarity",
                        "context_score",
                        "win_rate",
                        "regret_rate",
                        "stale_cost",
                        "sample_count",
                    }
                }

                combined_score = base_score
                combined_score *= bias_map.get(source, 1.0)
                if self.enable_reliability_bias:
                    reliability_score = 1.0 + WIN_WEIGHT * win_rate - REGRET_WEIGHT * regret_rate
                    reliability_score *= math.exp(-STALE_COST * stale_cost)
                    combined_score *= reliability_score
                else:
                    reliability_score = 1.0

                model_score = (
                    self._model_predict(source, feats) if self.use_ranker else 1.0
                )
                combined_score *= model_score

                severity = 0.0
                try:
                    if isinstance(item, dict):
                        severity = float(
                            item.get("alignment_severity")
                            or feats.get("alignment_severity")
                            or 0.0
                        )
                    else:
                        severity = float(
                            getattr(item, "alignment_severity", 0.0)
                            or feats.get("alignment_severity", 0.0)
                        )
                except Exception:
                    severity = 0.0
                if severity:
                    combined_score /= 1.0 + severity
                metrics["alignment_severity"] = severity

                scored.append(
                    {
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
                        "sample_count": sample_count,
                        "reliability_score": reliability_score,
                        "model_score": model_score,
                        **metrics,
                    }
                )

        _score_results(raw_results)
        scored.sort(key=lambda e: e["confidence"], reverse=True)

        if (
            self.fallback_on_low_reliability
            and self.reliability_threshold
            and self._reliability_stats
        ):
            top_rels = [
                self._reliability_stats.get(entry["source"], {}).get("reliability", 0.0)
                for entry in scored[:top_k]
            ]
            if top_rels and all(r < self.reliability_threshold for r in top_rels):
                high_rel_dbs = [
                    name
                    for name, stats in self._reliability_stats.items()
                    if stats.get("reliability", 0.0) >= self.reliability_threshold
                ]
                if high_rel_dbs:
                    fb_results = self._retrieve_candidates(
                        query, top_k, db_names=high_rel_dbs
                    )
                    fb_times = getattr(self, "_last_db_times", {})
                    db_times.update(fb_times)
                    fb_sources.extend(getattr(self, "_last_fallback_sources", []))
                    _score_results(fb_results)
                    scored.sort(key=lambda e: e["confidence"], reverse=True)

        self._last_db_times = db_times
        self._last_fallback_sources = list(dict.fromkeys(fb_sources))

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
            metrics.pop("exec_freq", None)
            metrics.pop("prior_hits", None)
            metrics.pop("roi_delta", None)
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
            if self._last_fallback_sources:
                meta["fallback_sources"] = list(self._last_fallback_sources)
            text = str(meta.get("text") or "")
            governed = govern_retrieval(text, meta, reason)
            if governed is None:
                continue
            meta, reason = governed
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
            context_score = float(bundle.metadata.get("context_score", 0.0))
            created_at = bundle.metadata.get("created_at")
            age = 0.0
            if created_at:
                try:
                    age = (datetime.utcnow() - datetime.fromisoformat(created_at)).total_seconds()
                except Exception:
                    age = 0.0
            win_rate = float(bundle.metadata.get("win_rate", 0.0))
            regret_rate = float(bundle.metadata.get("regret_rate", 0.0))
            ctx_metrics = bundle.metadata.get("contextual_metrics", {}) or {}
            frequency = ctx_metrics.get("frequency")
            roi_delta = ctx_metrics.get("roi")
            usage = ctx_metrics.get("usage")
            reliability_score = ctx_metrics.get("reliability_score")
            sample_count = ctx_metrics.get("sample_count")
            db_type = bundle.metadata.get("db_type") or bundle.metadata.get("db_source") or ""
            prior_hits = _prior_hit_count(bundle.origin_db, bundle.record_id)
            log_retrieval_metrics(
                bundle.origin_db,
                bundle.record_id,
                rank,
                included,
                tokens,
                similarity,
                context_score,
                age,
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
                "vector_id": str(bundle.record_id),
                "db_type": db_type,
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
                "age": age,
                "frequency": frequency,
                "roi_delta": roi_delta,
                "usage": usage,
                "prior_hits": prior_hits,
                "reliability_score": reliability_score,
                "sample_count": sample_count,
            }
            metrics_list.append(metrics_entry)
            dataset_entries.append(
                {
                    "origin_db": bundle.origin_db,
                    "db_type": db_type,
                    "record_id": bundle.record_id,
                    "vector_id": str(bundle.record_id),
                    "rank": rank,
                    "score": bundle.score,
                    "win_rate": win_rate,
                    "regret_rate": regret_rate,
                    "hit": included,
                    "session_id": session_id,
                    "age": age,
                    "similarity": similarity,
                    "context_score": context_score,
                    "frequency": frequency,
                    "roi_delta": roi_delta,
                    "usage": usage,
                    "prior_hits": prior_hits,
                    "reliability_score": reliability_score,
                    "sample_count": sample_count,
                }
            )

        results = hits[:top_k]
        vector_info: List[Tuple[str, str, float]] = []
        for h in results:
            vid = ""
            try:
                vid = str(h.metadata.get("vector_id"))  # type: ignore[union-attr]
            except Exception:
                vid = ""
            if not vid:
                vid = str(h.record_id)
            vector_info.append((h.origin_db, vid, float(h.score)))
        result_container = RetrievalResult(
            results, session_id, vector_info, self._last_fallback_sources
        )

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

        dataset_path = resolve_path("analytics/retrieval_outcomes.jsonl")
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
            return result_container, session_id, vector_info, metrics_list
        return result_container, session_id, vector_info

    # Backwards compatibility for older callers
    def retrieve_with_confidence(
        self,
        query: Any,
        top_k: int = 10,
        link_multiplier: float = 1.1,
        return_metrics: bool = False,
        roi_tracker: "ROITracker | None" = None,
        adjust_weights: bool = False,
    ) -> Union[
        Tuple["RetrievalResult", str, List[Tuple[str, str]]],
        Tuple["RetrievalResult", str, List[Tuple[str, str]], List[dict[str, Any]]],
    ]:
        res = self.retrieve(
            query,
            top_k=top_k,
            link_multiplier=link_multiplier,
            return_metrics=return_metrics,
            roi_tracker=roi_tracker,
            adjust_weights=adjust_weights,
        )
        if return_metrics:
            hits, session_id, vectors, metrics_list = res
        else:
            hits, session_id, vectors = res
            metrics_list = []
        formatted = [
            {
                "source": h.origin_db,
                "record_id": h.record_id,
                "item": h.metadata,
                "confidence": h.score,
                "reason": h.reason,
                "license": h.metadata.get("license"),
                "license_fingerprint": h.metadata.get("license_fingerprint"),
                "semantic_alerts": h.metadata.get("semantic_alerts"),
                "alignment_severity": h.metadata.get("alignment_severity"),
            }
            for h in hits
        ]
        result_container = RetrievalResult(
            formatted, session_id, vectors, getattr(hits, "fallback_sources", [])
        )
        if return_metrics:
            return result_container, session_id, vectors, metrics_list
        return result_container, session_id, vectors
