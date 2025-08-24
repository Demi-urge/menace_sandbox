from __future__ import annotations

"""Persist outcomes of experiment runs."""

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

from db_router import GLOBAL_ROUTER, init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope

MENACE_ID = "experiment_history_db"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)


@dataclass
class TestLog:
    """Record statistics from an A/B test comparison."""

    variant_a: str
    variant_b: str
    t_stat: float
    p_value: float
    ts: str = datetime.utcnow().isoformat()


@dataclass
class ExperimentLog:
    """Single experiment result."""

    variant: str
    roi: float
    cpu: float
    memory: float
    ts: str = datetime.utcnow().isoformat()


class ExperimentHistoryDB:
    """SQLite-backed store for experiment outcomes routed via ``DBRouter``."""

    def __init__(self) -> None:
        self.router = DB_ROUTER
        conn = self.router.get_connection("experiment_history")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_history(
                variant TEXT,
                roi REAL,
                cpu REAL,
                memory REAL,
                ts TEXT
            )
            """
        )
        conn.commit()
        conn = self.router.get_connection("experiment_tests")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_tests(
                variant_a TEXT,
                variant_b TEXT,
                t_stat REAL,
                p_value REAL,
                ts TEXT
            )
            """
        )
        conn.commit()

    def add(self, rec: ExperimentLog) -> None:
        conn = self.router.get_connection("experiment_history")
        conn.execute(
            "INSERT INTO experiment_history(variant, roi, cpu, memory, ts) VALUES(?,?,?,?,?)",
            (rec.variant, rec.roi, rec.cpu, rec.memory, rec.ts),
        )
        conn.commit()

    def add_test(self, rec: TestLog) -> None:
        conn = self.router.get_connection("experiment_tests")
        conn.execute(
            (
                "INSERT INTO experiment_tests(variant_a, variant_b, t_stat, p_value, ts)"
                " VALUES(?,?,?,?,?)"
            ),
            (rec.variant_a, rec.variant_b, rec.t_stat, rec.p_value, rec.ts),
        )
        conn.commit()

    def fetch(
        self, limit: int = 50, *, scope: Scope | str = "local"
    ) -> List[Tuple[str, float, float, float, str]]:
        """Return experiment logs filtered by menace ``scope``."""

        conn = self.router.get_connection("experiment_history")
        cur = conn.cursor()
        base = (
            "SELECT variant, roi, cpu, memory, ts FROM experiment_history"
        )
        clause, scope_params = build_scope_clause(
            "experiment_history", scope, self.router.menace_id
        )
        base = apply_scope(base, clause)
        base += " ORDER BY ts DESC LIMIT ?"
        params = [*scope_params, limit]
        cur.execute(base, params)
        return cur.fetchall()

    def variant_stats(self, variant: str) -> Tuple[int, float]:
        conn = self.router.get_connection("experiment_history")
        cur = conn.execute(
            "SELECT roi FROM experiment_history WHERE variant=?",
            (variant,),
        )
        rows = [float(r[0]) for r in cur.fetchall()]
        n = len(rows)
        var = statistics.variance(rows) if n > 1 else 0.0
        return n, var

    def variant_values(self, variant: str) -> List[float]:
        conn = self.router.get_connection("experiment_history")
        cur = conn.execute(
            "SELECT roi FROM experiment_history WHERE variant=?",
            (variant,),
        )
        return [float(r[0]) for r in cur.fetchall()]
