from __future__ import annotations

"""Persist evaluation history and compute deployment weights."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router

# Ensure router recognises our tables
LOCAL_TABLES.update({"evaluation_history", "weight_override"})


@dataclass
class EvaluationRecord:
    """Single evaluation result."""

    engine: str
    cv_score: float
    passed: bool = True
    error: str = ""
    ts: str = datetime.utcnow().isoformat()


class EvaluationHistoryDB:
    """SQLite-backed store of :class:`EvaluationRecord` values."""

    def __init__(self, *, router: DBRouter | None = None) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router("evaluation_history")
        self.conn = self.router.get_connection("evaluation_history")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_history(
                engine TEXT,
                cv_score REAL,
                passed INTEGER,
                error TEXT,
                ts TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weight_override(
                engine TEXT PRIMARY KEY,
                weight REAL
            )
            """
        )
        self.conn.commit()

    # ----------------------------------------------------------
    def add(self, rec: EvaluationRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO evaluation_history(engine, cv_score, passed, error, ts) VALUES(?,?,?,?,?)",
            (rec.engine, rec.cv_score, int(rec.passed), rec.error, rec.ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ----------------------------------------------------------
    def history(self, engine: str, limit: int = 50) -> List[Tuple[float, str, int, str]]:
        cur = self.conn.execute(
            "SELECT cv_score, ts, passed, error FROM evaluation_history WHERE engine=? ORDER BY ts DESC LIMIT ?",
            (engine, limit),
        )
        return [
            (float(r[0]), r[1], int(r[2]), str(r[3]))
            for r in cur.fetchall()
        ]

    # ----------------------------------------------------------
    def engines(self) -> Iterable[str]:
        cur = self.conn.execute(
            "SELECT DISTINCT engine FROM evaluation_history"
        )
        return [r[0] for r in cur.fetchall()]

    # ----------------------------------------------------------
    def deployment_weights(self) -> Dict[str, float]:
        cur = self.conn.execute(
            "SELECT engine, cv_score FROM evaluation_history WHERE passed=1"
        )
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for name, score in cur.fetchall():
            totals[name] = totals.get(name, 0.0) + float(score)
            counts[name] = counts.get(name, 0) + 1
        if not totals:
            return {}
        max_avg = max(totals[n] / counts[n] for n in totals)
        weights = {n: (totals[n] / counts[n]) / max_avg for n in totals}
        # apply overrides
        for eng, val in self.weight_overrides().items():
            weights[eng] = val
        return weights

    # ----------------------------------------------------------
    def set_weight(self, engine: str, weight: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO weight_override(engine, weight) VALUES(?,?)",
            (engine, weight),
        )
        self.conn.commit()

    # ----------------------------------------------------------
    def weight_overrides(self) -> Dict[str, float]:
        cur = self.conn.execute("SELECT engine, weight FROM weight_override")
        return {r[0]: float(r[1]) for r in cur.fetchall()}


__all__ = ["EvaluationRecord", "EvaluationHistoryDB"]
