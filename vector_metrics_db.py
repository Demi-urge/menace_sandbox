from __future__ import annotations

"""Lightweight SQLite store for vector operation metrics."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Mapping, Sequence
import json
import logging
import time

from db_router import GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from dynamic_path_router import resolve_path, get_project_root

try:  # pragma: no cover - optional dependency
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - fallback when running directly
    import metrics_exporter as _me  # type: ignore

# Prometheus gauges/counters
_EMBEDDING_TOKENS_TOTAL = _me.Gauge(
    "embedding_tokens_total",
    "Total tokens processed for embeddings",
)
_RETRIEVAL_HIT_RATE = _me.Gauge(
    "retrieval_hit_rate",
    "Fraction of retrieval results included in final prompt",
)
_RETRIEVER_WIN_RATE = getattr(_me, "retriever_win_rate", _me.Gauge(
    "retriever_win_rate",
    "Win rate of retrieval operations by database",
    ["db"],
))
_RETRIEVER_REGRET_RATE = getattr(_me, "retriever_regret_rate", _me.Gauge(
    "retriever_regret_rate",
    "Regret rate of retrieval operations by database",
    ["db"],
))


logger = logging.getLogger(__name__)


def _timestamp_payload(start: float | None = None, **extra: Any) -> Dict[str, Any]:
    payload = {"ts": datetime.utcnow().isoformat(), **extra}
    if start is not None:
        payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return payload


@dataclass
class VectorMetric:
    """Single vector operation metric record."""

    event_type: str
    db: str
    tokens: int = 0
    wall_time_ms: float = 0.0
    store_time_ms: float = 0.0
    hit: bool | None = None
    rank: int | None = None
    contribution: float | None = None
    prompt_tokens: int | None = None
    patch_id: str = ""
    session_id: str = ""
    vector_id: str = ""
    similarity: float | None = None
    context_score: float | None = None
    age: float | None = None
    win: bool | None = None
    regret: bool | None = None
    ts: str = datetime.utcnow().isoformat()


def default_vector_metrics_path(*, ensure_exists: bool = True) -> Path:
    """Return the default location for the vector metrics database.

    The original implementation relied on :func:`resolve_path` which raises a
    :class:`FileNotFoundError` when ``vector_metrics.db`` has not been created
    yet.  Some entry points instantiate :class:`VectorMetricsDB` (or import
    modules that resolve the default path) before the database exists, causing
    start-up to abort.  This helper mirrors the previous behaviour when the
    file is present while providing a deterministic fallback rooted at the
    repository when it is missing.

    When ``ensure_exists`` is ``True`` the parent directory is created and an
    empty file is touched so subsequent :func:`resolve_path` calls succeed.
    ``sqlite3`` will initialise the actual database schema on first use.
    """

    try:
        path = resolve_path("vector_metrics.db")
    except FileNotFoundError:
        path = (get_project_root() / "vector_metrics.db").resolve()
        if ensure_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
    else:
        if ensure_exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
    return path


class VectorMetricsDB:
    """SQLite-backed store for :class:`VectorMetric` records."""

    def __init__(self, path: Path | str = "vector_metrics.db") -> None:
        init_start = time.perf_counter()
        logger.info(
            "vector_metrics_db.init.start",
            extra=_timestamp_payload(init_start, configured_path=str(path)),
        )

        LOCAL_TABLES.add("vector_metrics")

        default_path = default_vector_metrics_path()
        requested = Path(path).expanduser()
        if str(requested.as_posix()) == "vector_metrics.db":
            p = default_path
        else:
            if not requested.is_absolute():
                requested = (default_path.parent / requested).resolve()
            else:
                requested = requested.resolve()
            requested.parent.mkdir(parents=True, exist_ok=True)
            p = requested

        logger.info(
            "vector_metrics_db.path.resolved",
            extra=_timestamp_payload(
                init_start, resolved_path=str(p), default_path=str(default_path)
            ),
        )

        if GLOBAL_ROUTER is not None and p == default_path:
            self.router = GLOBAL_ROUTER
            using_global_router = True
        else:
            self.router = init_db_router("vector_metrics_db", str(p), str(p))
            using_global_router = False
        wal = p.with_suffix(p.suffix + "-wal")
        shm = p.with_suffix(p.suffix + "-shm")
        for sidecar in (wal, shm):
            try:
                if sidecar.exists():
                    logger.warning(
                        "vector_metrics_db.sidecar.present",
                        extra=_timestamp_payload(
                            init_start,
                            sidecar=str(sidecar),
                            size=sidecar.stat().st_size,
                        ),
                    )
            except Exception:  # pragma: no cover - best effort diagnostics
                logger.exception("vector_metrics_db.sidecar.inspect_failed")

        self.conn = self.router.get_connection("vector_metrics")
        logger.info(
            "vector_metrics_db.connection.ready",
            extra=_timestamp_payload(
                init_start, using_global_router=using_global_router
            ),
        )
        schema_start = time.perf_counter()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_metrics(
                event_type TEXT,
                db TEXT,
                tokens INTEGER,
                wall_time_ms REAL,
                store_time_ms REAL,
                hit INTEGER,
                rank INTEGER,
                contribution REAL,
                prompt_tokens INTEGER,
                patch_id TEXT,
                session_id TEXT,
                vector_id TEXT,
                similarity REAL,
                context_score REAL,
                age REAL,
                win INTEGER,
                regret INTEGER,
                ts TEXT
            )
            """
        )
        logger.info(
            "vector_metrics_db.schema.ensured",
            extra=_timestamp_payload(schema_start),
        )
        migration_start = time.perf_counter()
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS vector_metrics_event_db_ts
                ON vector_metrics(event_type, db, ts)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_ancestry(
                patch_id TEXT,
                vector_id TEXT,
                rank INTEGER,
                contribution REAL,
                license TEXT,
                semantic_alerts TEXT,
                alignment_severity REAL,
                risk_score REAL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_metrics(
                patch_id TEXT PRIMARY KEY,
                errors TEXT,
                tests_passed INTEGER,
                lines_changed INTEGER,
                context_tokens INTEGER,
                patch_difficulty INTEGER,
                start_time REAL,
                time_to_completion REAL,
                error_trace_count INTEGER,
                roi_tag TEXT,
                effort_estimate REAL,
                enhancement_score REAL
            )
            """
        )
        # Store adaptive ranking weights so the ranker can learn over time.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ranking_weights(
                db TEXT PRIMARY KEY,
                weight REAL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_weights(
                vector_id TEXT PRIMARY KEY,
                weight REAL
            )
            """
        )
        # Persist session vector data so retrievals can be reconciled after
        # restarts.  Stored as JSON blobs keyed by ``session_id``.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_sessions(
                session_id TEXT PRIMARY KEY,
                vectors TEXT,
                metadata TEXT
            )
            """
        )
        self.conn.commit()
        cols = self._table_columns("vector_metrics")
        migrations = {
            "session_id": "ALTER TABLE vector_metrics ADD COLUMN session_id TEXT",
            "vector_id": "ALTER TABLE vector_metrics ADD COLUMN vector_id TEXT",
            "similarity": "ALTER TABLE vector_metrics ADD COLUMN similarity REAL",
            "context_score": "ALTER TABLE vector_metrics ADD COLUMN context_score REAL",
            "age": "ALTER TABLE vector_metrics ADD COLUMN age REAL",
            "win": "ALTER TABLE vector_metrics ADD COLUMN win INTEGER",
            "regret": "ALTER TABLE vector_metrics ADD COLUMN regret INTEGER",
        }
        applied_columns = []
        for name, stmt in migrations.items():
            if name not in cols:
                self.conn.execute(stmt)
                applied_columns.append(name)
        logger.info(
            "vector_metrics_db.migrations.vector_metrics",
            extra=_timestamp_payload(
                migration_start, applied_columns=applied_columns
            ),
        )
        self.conn.commit()
        pcols = self._table_columns("patch_ancestry")
        if "license" not in pcols:
            self.conn.execute("ALTER TABLE patch_ancestry ADD COLUMN license TEXT")
        if "semantic_alerts" not in pcols:
            self.conn.execute("ALTER TABLE patch_ancestry ADD COLUMN semantic_alerts TEXT")
        if "alignment_severity" not in pcols:
            self.conn.execute(
                "ALTER TABLE patch_ancestry ADD COLUMN alignment_severity REAL"
            )
        if "risk_score" not in pcols:
            self.conn.execute(
                "ALTER TABLE patch_ancestry ADD COLUMN risk_score REAL"
            )
        self.conn.commit()
        mcols = self._table_columns("patch_metrics")
        if "context_tokens" not in mcols:
            self.conn.execute("ALTER TABLE patch_metrics ADD COLUMN context_tokens INTEGER")
        if "patch_difficulty" not in mcols:
            self.conn.execute("ALTER TABLE patch_metrics ADD COLUMN patch_difficulty INTEGER")
        if "start_time" not in mcols:
            self.conn.execute("ALTER TABLE patch_metrics ADD COLUMN start_time REAL")
        if "time_to_completion" not in mcols:
            self.conn.execute(
                "ALTER TABLE patch_metrics ADD COLUMN time_to_completion REAL"
            )
        if "error_trace_count" not in mcols:
            self.conn.execute(
                "ALTER TABLE patch_metrics ADD COLUMN error_trace_count INTEGER"
            )
        if "roi_tag" not in mcols:
            self.conn.execute("ALTER TABLE patch_metrics ADD COLUMN roi_tag TEXT")
        if "effort_estimate" not in mcols:
            self.conn.execute("ALTER TABLE patch_metrics ADD COLUMN effort_estimate REAL")
        if "enhancement_score" not in mcols:
            self.conn.execute(
                "ALTER TABLE patch_metrics ADD COLUMN enhancement_score REAL"
            )
        self.conn.commit()
        logger.info(
            "vector_metrics_db.migrations.patch_tables",
            extra=_timestamp_payload(
                migration_start,
                patch_ancestry_missing=[
                    c
                    for c in (
                        "license",
                        "semantic_alerts",
                        "alignment_severity",
                        "risk_score",
                    )
                    if c not in pcols
                ],
                patch_metrics_missing=[
                    c
                    for c in (
                        "context_tokens",
                        "patch_difficulty",
                        "start_time",
                        "time_to_completion",
                        "error_trace_count",
                        "roi_tag",
                        "effort_estimate",
                        "enhancement_score",
                    )
                    if c not in mcols
                ],
            ),
        )
        logger.info(
            "vector_metrics_db.init.complete",
            extra=_timestamp_payload(
                init_start,
                resolved_path=str(p),
                using_global_router=using_global_router,
            ),
        )

    def _table_columns(self, table: str) -> list[str]:
        """Return column names for ``table`` using non-blocking pragmas."""

        start = time.perf_counter()
        try:
            rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        except Exception:
            logger.exception(
                "vector_metrics_db.schema.inspect_failed",
                extra=_timestamp_payload(start, table=table),
            )
            return []
        logger.info(
            "vector_metrics_db.schema.inspected",
            extra=_timestamp_payload(start, table=table, column_count=len(rows)),
        )
        return [r[1] for r in rows]

    # ------------------------------------------------------------------
    def get_db_weights(self) -> dict[str, float]:
        """Return mapping of origin database to current ranking weight."""

        cur = self.conn.execute("SELECT db, weight FROM ranking_weights")
        rows = cur.fetchall()
        return {str(db): float(weight) for db, weight in rows}

    # ------------------------------------------------------------------
    def update_db_weight(self, db: str, delta: float, *, normalize: bool = False) -> float:
        """Adjust ranking weight for *db* by ``delta`` and persist it.

        Weights are clamped to the inclusive range ``[0, 1]`` so repeated
        positive or negative feedback cannot push them outside sensible
        bounds.  When ``normalize`` is ``True`` all weights are also
        renormalised so their sum equals 1.0.  The new weight for ``db`` is
        returned (after optional normalisation)."""

        cur = self.conn.execute(
            "SELECT weight FROM ranking_weights WHERE db=?", (db,)
        )
        row = cur.fetchone()
        weight = float(row[0]) if row and row[0] is not None else 0.0
        weight = max(0.0, min(1.0, weight + delta))
        self.conn.execute(
            "REPLACE INTO ranking_weights(db, weight) VALUES(?, ?)", (db, weight)
        )
        self.conn.commit()
        if normalize:
            return self.normalize_db_weights().get(db, weight)
        return weight

    # ------------------------------------------------------------------
    def normalize_db_weights(self) -> dict[str, float]:
        """Scale all weights so they sum to 1.0.

        Returns the normalised weight mapping."""

        cur = self.conn.execute("SELECT db, weight FROM ranking_weights")
        rows = [(str(db), float(w)) for db, w in cur.fetchall()]
        total = sum(w for _db, w in rows)
        if total > 0:
            for db, w in rows:
                norm = w / total
                self.conn.execute(
                    "REPLACE INTO ranking_weights(db, weight) VALUES(?, ?)",
                    (db, norm),
                )
            self.conn.commit()
            return {db: w / total for db, w in rows}
        return {db: w for db, w in rows}

    # ------------------------------------------------------------------
    def set_db_weights(self, weights: Dict[str, float]) -> None:
        """Persist full ranking weight mapping."""

        rows = [
            (str(db), max(0.0, min(1.0, float(w)))) for db, w in weights.items()
        ]
        self.conn.executemany(
            "REPLACE INTO ranking_weights(db, weight) VALUES(?, ?)", rows
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def update_vector_weight(self, vector_id: str, delta: float) -> float:
        """Adjust ranking weight for *vector_id* by ``delta`` and persist it."""

        cur = self.conn.execute(
            "SELECT weight FROM vector_weights WHERE vector_id=?", (vector_id,)
        )
        row = cur.fetchone()
        weight = float(row[0]) if row and row[0] is not None else 0.0
        weight = max(0.0, min(1.0, weight + delta))
        self.conn.execute(
            "REPLACE INTO vector_weights(vector_id, weight) VALUES(?, ?)",
            (vector_id, weight),
        )
        self.conn.commit()
        return weight

    # ------------------------------------------------------------------
    def get_vector_weight(self, vector_id: str) -> float:
        """Return ranking weight for *vector_id* (0.0 if unknown)."""

        cur = self.conn.execute(
            "SELECT weight FROM vector_weights WHERE vector_id=?", (vector_id,)
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    # ------------------------------------------------------------------
    def set_vector_weight(self, vector_id: str, weight: float) -> None:
        """Persist absolute weight value for *vector_id*."""

        weight = max(0.0, min(1.0, float(weight)))
        self.conn.execute(
            "REPLACE INTO vector_weights(vector_id, weight) VALUES(?, ?)",
            (vector_id, weight),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def recalc_ranking_weights(self) -> Dict[str, float]:
        """Recalculate ranking weights from cumulative ROI and safety data."""

        cur = self.conn.execute(
            """
            SELECT db,
                   COALESCE(SUM(contribution),0) AS roi,
                   COALESCE(AVG(win),0) AS win_rate,
                   COALESCE(AVG(regret),0) AS regret_rate
              FROM vector_metrics
             WHERE event_type='retrieval'
             GROUP BY db
            """
        )
        weights: Dict[str, float] = {}
        for db, roi, win_rate, regret_rate in cur.fetchall():
            roi = float(roi or 0.0)
            win = float(win_rate or 0.0)
            regret = float(regret_rate or 0.0)
            score = roi * max(win, 0.01) * (1.0 - regret)
            if score < 0:
                score = 0.0
            weights[str(db)] = score
        total = sum(weights.values())
        if total > 0:
            weights = {db: w / total for db, w in weights.items()}
        self.set_db_weights(weights)
        return weights

    # ------------------------------------------------------------------
    def save_session(
        self,
        session_id: str,
        vectors: List[Tuple[str, str, float]],
        metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Persist session retrieval data for later reconciliation."""

        self.conn.execute(
            "REPLACE INTO pending_sessions(session_id, vectors, metadata) VALUES(?,?,?)",
            (session_id, json.dumps(vectors), json.dumps(metadata)),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def load_sessions(
        self,
    ) -> Dict[str, Tuple[List[Tuple[str, str, float]], Dict[str, Dict[str, Any]]]]:
        """Return mapping of session_id to stored vectors and metadata."""

        cur = self.conn.execute(
            "SELECT session_id, vectors, metadata FROM pending_sessions"
        )
        sessions: Dict[str, Tuple[List[Tuple[str, str, float]], Dict[str, Dict[str, Any]]]] = {}
        for sid, vec_json, meta_json in cur.fetchall():
            try:
                raw_vecs = json.loads(vec_json or "[]")
                vecs = [
                    (str(o), str(v), float(s))
                    for o, v, s in raw_vecs
                ]
            except Exception:
                vecs = []
            try:
                meta = json.loads(meta_json or "{}")
            except Exception:
                meta = {}
            sessions[str(sid)] = (vecs, meta)
        return sessions

    # ------------------------------------------------------------------
    def delete_session(self, session_id: str) -> None:
        """Remove persisted session data once outcome recorded."""

        self.conn.execute(
            "DELETE FROM pending_sessions WHERE session_id=?",
            (session_id,),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def add(self, rec: VectorMetric) -> None:
        self.conn.execute(
            """
            INSERT INTO vector_metrics(
                event_type, db, tokens, wall_time_ms, store_time_ms, hit,
                rank, contribution, prompt_tokens, patch_id, session_id,
                vector_id, similarity, context_score, age, win, regret, ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.event_type,
                rec.db,
                int(rec.tokens),
                float(rec.wall_time_ms),
                float(rec.store_time_ms),
                None if rec.hit is None else int(rec.hit),
                rec.rank,
                rec.contribution,
                rec.prompt_tokens,
                rec.patch_id,
                rec.session_id,
                rec.vector_id,
                rec.similarity,
                rec.context_score,
                rec.age,
                None if rec.win is None else int(rec.win),
                None if rec.regret is None else int(rec.regret),
                rec.ts,
            ),
        )
        self.conn.commit()
        if rec.event_type == "embedding":
            try:  # best-effort metrics
                _EMBEDDING_TOKENS_TOTAL.inc(rec.tokens)
            except Exception:
                pass
        elif rec.event_type == "retrieval":
            self._update_retrieval_hit_rate()

    # ------------------------------------------------------------------
    def log_embedding(
        self,
        db: str,
        tokens: int,
        wall_time_ms: float,
        *,
        store_time_ms: float = 0.0,
        prompt_tokens: int | None = None,
        patch_id: str = "",
        vector_id: str = "",
    ) -> None:
        rec = VectorMetric(
            event_type="embedding",
            db=db,
            tokens=tokens,
            wall_time_ms=wall_time_ms,
            store_time_ms=store_time_ms,
            prompt_tokens=prompt_tokens,
            patch_id=patch_id,
            vector_id=vector_id,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def log_retrieval(
        self,
        db: str,
        tokens: int,
        wall_time_ms: float,
        *,
        hit: bool,
        rank: int,
        contribution: float = 0.0,
        prompt_tokens: int = 0,
        patch_id: str = "",
        store_time_ms: float = 0.0,
        session_id: str = "",
        vector_id: str = "",
        similarity: float = 0.0,
        context_score: float = 0.0,
        age: float = 0.0,
    ) -> None:
        rec = VectorMetric(
            event_type="retrieval",
            db=db,
            tokens=tokens,
            wall_time_ms=wall_time_ms,
            store_time_ms=store_time_ms,
            hit=hit,
            rank=rank,
            contribution=contribution,
            prompt_tokens=prompt_tokens,
            patch_id=patch_id,
            session_id=session_id,
            vector_id=vector_id,
            similarity=similarity,
            context_score=context_score,
            age=age,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def log_retrieval_feedback(
        self,
        db: str,
        *,
        win: bool = False,
        regret: bool = False,
        roi: float = 0.0,
    ) -> None:
        """Persist aggregate feedback for *db* without session context."""

        rec = VectorMetric(
            event_type="retrieval",
            db=db,
            tokens=0,
            wall_time_ms=0.0,
            contribution=roi,
            win=win,
            regret=regret,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def log_ranker_update(
        self, db: str, *, delta: float, weight: float | None = None
    ) -> None:
        """Record a ranking weight adjustment for ``db``.

        The ``delta`` reflects the change applied to the weight while
        ``weight`` captures the resulting value when available.  Entries are
        stored in :class:`VectorMetric` with ``event_type`` set to ``"ranker"``
        so historical adjustments can be analysed alongside other vector
        metrics.
        """

        rec = VectorMetric(
            event_type="ranker",
            db=db,
            tokens=0,
            wall_time_ms=0.0,
            contribution=delta,
            similarity=weight,
            context_score=weight,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def embedding_tokens_total(self, db: str | None = None) -> int:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(tokens),0) FROM vector_metrics WHERE event_type='embedding'" +
            (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return int(res[0] if res and res[0] is not None else 0)

    # ------------------------------------------------------------------
    def retrieval_hit_rate(self, db: str | None = None) -> float:
        cur = self.conn.execute(
            "SELECT AVG(hit) FROM vector_metrics WHERE event_type='retrieval'" +
            (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_win_rate(self, db: str | None = None) -> float:
        cur = self.conn.execute(
            "SELECT AVG(win) FROM vector_metrics "
            "WHERE event_type='retrieval' AND win IS NOT NULL"
            + (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_regret_rate(self, db: str | None = None) -> float:
        cur = self.conn.execute(
            "SELECT AVG(regret) FROM vector_metrics "
            "WHERE event_type='retrieval' AND regret IS NOT NULL"
            + (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_win_rate_by_db(self) -> dict[str, float]:
        cur = self.conn.execute(
            """
            SELECT db, AVG(win)
              FROM vector_metrics
             WHERE event_type='retrieval' AND win IS NOT NULL
             GROUP BY db
            """
        )
        rows = cur.fetchall()
        rates = {str(db): float(rate) if rate is not None else 0.0 for db, rate in rows}
        for name, rate in rates.items():
            try:
                _RETRIEVER_WIN_RATE.labels(db=name).set(rate)
            except Exception:
                pass
        return rates

    # ------------------------------------------------------------------
    def retriever_regret_rate_by_db(self) -> dict[str, float]:
        cur = self.conn.execute(
            """
            SELECT db, AVG(regret)
              FROM vector_metrics
             WHERE event_type='retrieval' AND regret IS NOT NULL
             GROUP BY db
            """
        )
        rows = cur.fetchall()
        rates = {str(db): float(rate) if rate is not None else 0.0 for db, rate in rows}
        for name, rate in rates.items():
            try:
                _RETRIEVER_REGRET_RATE.labels(db=name).set(rate)
            except Exception:
                pass
        return rates

    # ------------------------------------------------------------------
    def update_outcome(
        self,
        session_id: str,
        vectors: list[tuple[str, str]],
        *,
        contribution: float,
        patch_id: str = "",
        win: bool = False,
        regret: bool = False,
    ) -> None:
        for _, vec_id in vectors:
            self.conn.execute(
                """
                UPDATE vector_metrics
                   SET contribution=?, win=?, regret=?, patch_id=?
                 WHERE session_id=? AND vector_id=?
                """,
                (
                    contribution,
                    int(win),
                    int(regret),
                    patch_id,
                    session_id,
                    vec_id,
                ),
            )
        self.conn.commit()

    def record_patch_ancestry(
        self, patch_id: str, vectors: list[tuple]
    ) -> None:
        for rank, vec in enumerate(vectors):
            vec_id, contrib, lic, alerts, sev, risk = (
                list(vec) + [None, None, None, None, None]
            )[:6]
            self.conn.execute(
                "INSERT INTO patch_ancestry(patch_id, vector_id, rank, contribution, "
                "license, semantic_alerts, alignment_severity, risk_score) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    vec_id,
                    rank,
                    contrib,
                    lic,
                    json.dumps(alerts) if alerts is not None else None,
                    sev,
                    risk,
                ),
            )
        self.conn.commit()

    def record_patch_summary(
        self,
        patch_id: str,
        *,
        errors: Sequence[Mapping[str, Any]] | None = None,
        tests_passed: bool | None = None,
        lines_changed: int | None = None,
        context_tokens: int | None = None,
        patch_difficulty: int | None = None,
        start_time: float | None = None,
        time_to_completion: float | None = None,
        error_trace_count: int | None = None,
        roi_tag: str | None = None,
        effort_estimate: float | None = None,
        enhancement_score: float | None = None,
    ) -> None:
        try:
            self.conn.execute(
                "REPLACE INTO patch_metrics(patch_id, errors, tests_passed, "
                "lines_changed, context_tokens, patch_difficulty, start_time, "
                "time_to_completion, error_trace_count, roi_tag, "
                "effort_estimate, enhancement_score) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    json.dumps(list(errors or [])),
                    None if tests_passed is None else int(bool(tests_passed)),
                    lines_changed,
                    context_tokens,
                    patch_difficulty,
                    start_time,
                    time_to_completion,
                    error_trace_count,
                    roi_tag,
                    effort_estimate,
                    enhancement_score,
                ),
            )
            self.conn.commit()
        except Exception:
            logging.getLogger(__name__).exception("failed to record patch summary")

    # ------------------------------------------------------------------
    def _update_retrieval_hit_rate(self) -> None:
        try:  # best-effort metrics
            _RETRIEVAL_HIT_RATE.set(self.retrieval_hit_rate())
        except Exception:
            pass


__all__ = ["VectorMetric", "VectorMetricsDB"]
