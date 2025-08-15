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

    # ------------------------------------------------------------------
    def add(self, rec: VectorMetric) -> None:
        self.conn.execute(
            """
            INSERT INTO vector_metrics(
                event_type, db, tokens, wall_time_ms, store_time_ms, hit,
                rank, contribution, prompt_tokens, patch_id, ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
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
    ) -> None:
        rec = VectorMetric(
            event_type="embedding",
            db=db,
            tokens=tokens,
            wall_time_ms=wall_time_ms,
            store_time_ms=store_time_ms,
            prompt_tokens=prompt_tokens,
            patch_id=patch_id,
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
    def _update_retrieval_hit_rate(self) -> None:
        try:  # best-effort metrics
            _RETRIEVAL_HIT_RATE.set(self.retrieval_hit_rate())
        except Exception:
            pass


__all__ = ["VectorMetric", "VectorMetricsDB"]
