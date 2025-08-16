from __future__ import annotations

"""Lightweight SQLite store for vector operation metrics."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3

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


class VectorMetricsDB:
    """SQLite-backed store for :class:`VectorMetric` records."""

    def __init__(self, path: Path | str = "vector_metrics.db") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
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
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS vector_metrics_event_db_ts
                ON vector_metrics(event_type, db, ts)
            """
        )
        self.conn.commit()
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(vector_metrics)").fetchall()]
        migrations = {
            "session_id": "ALTER TABLE vector_metrics ADD COLUMN session_id TEXT",
            "vector_id": "ALTER TABLE vector_metrics ADD COLUMN vector_id TEXT",
            "similarity": "ALTER TABLE vector_metrics ADD COLUMN similarity REAL",
            "context_score": "ALTER TABLE vector_metrics ADD COLUMN context_score REAL",
            "age": "ALTER TABLE vector_metrics ADD COLUMN age REAL",
            "win": "ALTER TABLE vector_metrics ADD COLUMN win INTEGER",
            "regret": "ALTER TABLE vector_metrics ADD COLUMN regret INTEGER",
        }
        for name, stmt in migrations.items():
            if name not in cols:
                self.conn.execute(stmt)
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
            "SELECT AVG(win) FROM vector_metrics WHERE event_type='retrieval' AND win IS NOT NULL"
            + (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_regret_rate(self, db: str | None = None) -> float:
        cur = self.conn.execute(
            "SELECT AVG(regret) FROM vector_metrics WHERE event_type='retrieval' AND regret IS NOT NULL"
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

    # ------------------------------------------------------------------
    def _update_retrieval_hit_rate(self) -> None:
        try:  # best-effort metrics
            _RETRIEVAL_HIT_RATE.set(self.retrieval_hit_rate())
        except Exception:
            pass


__all__ = ["VectorMetric", "VectorMetricsDB"]
