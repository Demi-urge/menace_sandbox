"""Data Bot for collecting and analysing performance metrics.

The bot exposes a small set of configuration keys that control how
``ROI`` and error deltas are evaluated when looking for performance
degradation.  These values are typically provided by
``SandboxSettings`` but can also be overridden per bot via
``config/self_coding_thresholds.yaml``.

Configuration Keys
------------------
``self_coding_roi_drop``
    Maximum allowed decrease in ROI before a bot is considered degraded.
``self_coding_error_increase``
    Maximum allowed increase in error count before a bot is considered
    degraded.
``self_coding_test_failure_increase``
    Maximum allowed increase in failed test count before a bot is
    considered degraded.

``model``
    Forecast model to use for predicting future metrics (e.g.,
    ``exponential`` or ``arima``).
``confidence``
    Confidence level for the forecast model's interval.

Example ``self_coding_thresholds.yaml``::

    default:
      roi_drop: -0.1
      error_increase: 1.0
      test_failure_increase: 0.0
      model: exponential
      confidence: 0.95
    bots:
      example-bot:
        roi_drop: -0.2
        error_increase: 2.0
        test_failure_increase: 0.0
        model: arima
        confidence: 0.9
"""

from __future__ import annotations

# ``self_coding_managed`` is optional during import to allow lightweight tests
# that do not provide the full self-coding stack.  When unavailable the
# decorator becomes a no-op.
try:  # pragma: no cover - allow tests to stub missing dependencies
    from .coding_bot_interface import self_coding_managed
except Exception:  # pragma: no cover
    def self_coding_managed(cls):  # type: ignore[override]
        return cls
# flake8: noqa
import sqlite3
import os
import logging
import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, TYPE_CHECKING, Callable

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from .scope_utils import Scope, build_scope_clause, apply_scope

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover
    class UnifiedEventBus:  # type: ignore[override]
        pass
from .roi_thresholds import ROIThresholds
from .self_coding_thresholds import (
    get_thresholds as load_sc_thresholds,
    update_thresholds as save_sc_thresholds,
    adaptive_thresholds,
)
from .sandbox_settings import SandboxSettings
from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from .code_database import PatchHistoryDB
from .self_improvement.baseline_tracker import BaselineTracker

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .capital_management_bot import CapitalManagementBot

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore
    logging.getLogger(__name__).warning(
        "psutil is not installed; install with 'pip install psutil' to capture system metrics"
    )

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    logging.getLogger(__name__).warning(
        "pandas is not installed; install with 'pip install pandas' for DataFrame support"
    )

try:
    from prometheus_client import CollectorRegistry, Gauge  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CollectorRegistry = Gauge = None  # type: ignore
    logging.getLogger(__name__).warning(
        "prometheus_client is not installed; install with 'pip install prometheus-client' for Prometheus metrics"
    )

try:  # pragma: no cover - optional dependency
    from .vector_metrics_db import VectorMetricsDB
except Exception:
    VectorMetricsDB = None  # type: ignore

_VEC_METRICS = VectorMetricsDB() if VectorMetricsDB is not None else None


logger = logging.getLogger(__name__)
@dataclass
class MetricRecord:
    """Metrics captured for a bot at a point in time."""

    bot: str
    cpu: float
    memory: float
    response_time: float
    disk_io: float
    net_io: float
    errors: int
    tests_failed: int = 0
    tests_run: int = 0
    revenue: float = 0.0
    expense: float = 0.0
    security_score: float = 0.0
    safety_rating: float = 0.0
    adaptability: float = 0.0
    antifragility: float = 0.0
    shannon_entropy: float = 0.0
    efficiency: float = 0.0
    flexibility: float = 0.0
    gpu_usage: float = 0.0
    projected_lucrativity: float = 0.0
    profitability: float = 0.0
    patch_complexity: float = 0.0
    patch_entropy: float = 0.0
    energy_consumption: float = 0.0
    resilience: float = 0.0
    network_latency: float = 0.0
    throughput: float = 0.0
    risk_index: float = 0.0
    maintainability: float = 0.0
    code_quality: float = 0.0
    ts: str = datetime.utcnow().isoformat()


class MetricsDB:
    """SQLite-backed storage for metrics."""
    def __init__(
        self,
        path: Path | str = "metrics.db",
        router: DBRouter | None = None,
    ) -> None:
        self.path = str(path)
        LOCAL_TABLES.add("metrics")
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "metrics_db", local_db_path=self.path, shared_db_path=self.path
        )
        conn = self.router.get_connection("metrics")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot TEXT,
                cpu REAL,
                memory REAL,
                response_time REAL,
                disk_io REAL,
                net_io REAL,
                errors INTEGER,
                tests_failed INTEGER DEFAULT 0,
                tests_run INTEGER DEFAULT 0,
                revenue REAL DEFAULT 0,
                expense REAL DEFAULT 0,
                security_score REAL DEFAULT 0,
                safety_rating REAL DEFAULT 0,
                adaptability REAL DEFAULT 0,
                antifragility REAL DEFAULT 0,
                shannon_entropy REAL DEFAULT 0,
                efficiency REAL DEFAULT 0,
                flexibility REAL DEFAULT 0,
                gpu_usage REAL DEFAULT 0,
                projected_lucrativity REAL DEFAULT 0,
                profitability REAL DEFAULT 0,
                patch_complexity REAL DEFAULT 0,
                patch_entropy REAL DEFAULT 0,
                energy_consumption REAL DEFAULT 0,
                resilience REAL DEFAULT 0,
                network_latency REAL DEFAULT 0,
                throughput REAL DEFAULT 0,
                risk_index REAL DEFAULT 0,
                maintainability REAL DEFAULT 0,
                code_quality REAL DEFAULT 0,
                ts TEXT,
                source_menace_id TEXT NOT NULL
            )
            """
            )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS eval_metrics(
            cycle TEXT,
            metric TEXT,
            value REAL,
            ts TEXT,
            source_menace_id TEXT NOT NULL
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS embedding_metrics(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT,
            tokens INTEGER,
            wall_time REAL,
            index_latency REAL,
            source TEXT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS embedding_staleness(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_db TEXT,
            record_id TEXT,
            stale_seconds REAL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS retrieval_metrics(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_db TEXT,
            record_id TEXT,
            rank INTEGER,
            hit INTEGER,
            tokens INTEGER,
            score REAL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS training_stats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            success INTEGER,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS retriever_kpi(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_db TEXT,
            win_rate REAL,
            regret_rate REAL,
            stale_penalty REAL,
            sample_count REAL DEFAULT 0,
            roi REAL DEFAULT 0,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS patch_outcomes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patch_id TEXT,
            session_id TEXT,
            origin_db TEXT,
            vector_id TEXT,
            success INTEGER,
            reverted INTEGER DEFAULT 0,
            label TEXT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS retriever_stats(
            origin_db TEXT PRIMARY KEY,
            wins INTEGER DEFAULT 0,
            regrets INTEGER DEFAULT 0
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS embedding_stats(
            db_name TEXT,
            tokens INTEGER,
            wall_ms REAL,
            store_ms REAL,
            patch_id TEXT,
            db_source TEXT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS retrieval_stats(
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
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_cycle ON eval_metrics(cycle)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_source ON eval_metrics(source_menace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_ts ON embedding_metrics(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_stale_ts ON embedding_staleness(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_ts ON retrieval_metrics(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retriever_kpi_ts ON retriever_kpi(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_outcomes_ts ON patch_outcomes(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_outcomes_origin ON patch_outcomes(origin_db)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_stats_ts ON embedding_stats(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_stats_ts ON retrieval_stats(ts)"
        )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(metrics)").fetchall()]
        if "tests_failed" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN tests_failed INTEGER DEFAULT 0"
            )
        if "tests_run" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN tests_run INTEGER DEFAULT 0"
            )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(embedding_stats)").fetchall()]
        if "patch_id" not in cols:
            conn.execute("ALTER TABLE embedding_stats ADD COLUMN patch_id TEXT")
        if "db_source" not in cols:
            conn.execute("ALTER TABLE embedding_stats ADD COLUMN db_source TEXT")
        if "ts" not in cols:
            conn.execute(
                "ALTER TABLE embedding_stats ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP"
            )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(retrieval_stats)").fetchall()]
        for column, stmt in {
            "patch_id": "ALTER TABLE retrieval_stats ADD COLUMN patch_id TEXT",
            "db_source": "ALTER TABLE retrieval_stats ADD COLUMN db_source TEXT",
            "hit_rate": "ALTER TABLE retrieval_stats ADD COLUMN hit_rate REAL",
            "tokens_injected": "ALTER TABLE retrieval_stats ADD COLUMN tokens_injected INTEGER",
            "contribution": "ALTER TABLE retrieval_stats ADD COLUMN contribution REAL",
            "vector_id": "ALTER TABLE retrieval_stats ADD COLUMN vector_id TEXT",
            "db_type": "ALTER TABLE retrieval_stats ADD COLUMN db_type TEXT",
            "age": "ALTER TABLE retrieval_stats ADD COLUMN age REAL",
            "similarity": "ALTER TABLE retrieval_stats ADD COLUMN similarity REAL",
            "frequency": "ALTER TABLE retrieval_stats ADD COLUMN frequency REAL",
            "roi_delta": "ALTER TABLE retrieval_stats ADD COLUMN roi_delta REAL",
            "usage": "ALTER TABLE retrieval_stats ADD COLUMN usage REAL",
            "prior_hits": "ALTER TABLE retrieval_stats ADD COLUMN prior_hits INTEGER",
            "ts": "ALTER TABLE retrieval_stats ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP",
        }.items():
            if column not in cols:
                conn.execute(stmt)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(eval_metrics)").fetchall()]
        if "source_menace_id" not in cols:
            conn.execute(
                "ALTER TABLE eval_metrics ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
            )
            conn.execute(
                "UPDATE eval_metrics SET source_menace_id=? WHERE source_menace_id=''",
                (self.router.menace_id,),
            )
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(retriever_kpi)").fetchall()
        ]
        if "roi" not in cols:
            conn.execute(
                "ALTER TABLE retriever_kpi ADD COLUMN roi REAL DEFAULT 0"
            )
        if "sample_count" not in cols:
            conn.execute(
                "ALTER TABLE retriever_kpi ADD COLUMN sample_count REAL DEFAULT 0"
            )
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(embedding_metrics)").fetchall()
        ]
        if "ts" not in cols:
            conn.execute(
                "ALTER TABLE embedding_metrics ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP"
            )
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(retrieval_metrics)").fetchall()
        ]
        if "ts" not in cols:
            conn.execute(
                "ALTER TABLE retrieval_metrics ADD COLUMN ts TEXT DEFAULT CURRENT_TIMESTAMP"
            )
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(patch_outcomes)").fetchall()
        ]
        if "session_id" not in cols:
            conn.execute(
                "ALTER TABLE patch_outcomes ADD COLUMN session_id TEXT"
            )
        if "origin_db" not in cols:
            conn.execute(
                "ALTER TABLE patch_outcomes ADD COLUMN origin_db TEXT"
            )
        if "vector_id" not in cols:
            conn.execute(
                "ALTER TABLE patch_outcomes ADD COLUMN vector_id TEXT"
            )
        if "reverted" not in cols:
            conn.execute(
                "ALTER TABLE patch_outcomes ADD COLUMN reverted INTEGER DEFAULT 0",
            )
        if "label" not in cols:
            conn.execute(
                "ALTER TABLE patch_outcomes ADD COLUMN label TEXT",
            )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(metrics)").fetchall()]
        if "source_menace_id" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
            )
        if "revenue" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN revenue REAL DEFAULT 0")
        if "expense" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN expense REAL DEFAULT 0")
        if "security_score" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN security_score REAL DEFAULT 0")
        if "safety_rating" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN safety_rating REAL DEFAULT 0")
        if "adaptability" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN adaptability REAL DEFAULT 0")
        if "antifragility" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN antifragility REAL DEFAULT 0")
        if "shannon_entropy" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN shannon_entropy REAL DEFAULT 0")
        if "efficiency" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN efficiency REAL DEFAULT 0")
        if "flexibility" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN flexibility REAL DEFAULT 0")
        if "gpu_usage" not in cols:
            conn.execute("ALTER TABLE metrics ADD COLUMN gpu_usage REAL DEFAULT 0")
        if "projected_lucrativity" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN projected_lucrativity REAL DEFAULT 0"
            )
        if "profitability" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN profitability REAL DEFAULT 0"
            )
        if "patch_complexity" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN patch_complexity REAL DEFAULT 0"
            )
        if "patch_entropy" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN patch_entropy REAL DEFAULT 0"
            )
        if "energy_consumption" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN energy_consumption REAL DEFAULT 0"
            )
        if "resilience" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN resilience REAL DEFAULT 0"
            )
        if "network_latency" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN network_latency REAL DEFAULT 0"
            )
        if "throughput" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN throughput REAL DEFAULT 0"
            )
        if "risk_index" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN risk_index REAL DEFAULT 0"
            )
        if "maintainability" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN maintainability REAL DEFAULT 0"
            )
        if "code_quality" not in cols:
            conn.execute(
                "ALTER TABLE metrics ADD COLUMN code_quality REAL DEFAULT 0"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_bot ON metrics(bot)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_menace ON metrics(source_menace_id)"
        )
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return self.router.get_connection("metrics")

    def add(
        self, rec: MetricRecord, *, source_menace_id: object | None = None
    ) -> int:
        with self._connect() as conn:
            menace_id = source_menace_id or self.router.menace_id
            cur = conn.execute(
                """
                INSERT INTO metrics(
                    bot, cpu, memory, response_time, disk_io, net_io, errors,
                    tests_failed, tests_run, revenue, expense,
                    security_score, safety_rating, adaptability,
                    antifragility, shannon_entropy, efficiency,
                    flexibility, gpu_usage, projected_lucrativity,
                    profitability, patch_complexity, patch_entropy, energy_consumption,
                    resilience, network_latency, throughput, risk_index,
                    maintainability, code_quality, ts, source_menace_id)
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    rec.bot,
                    rec.cpu,
                    rec.memory,
                    rec.response_time,
                    rec.disk_io,
                    rec.net_io,
                    rec.errors,
                    rec.tests_failed,
                    rec.tests_run,
                    rec.revenue,
                    rec.expense,
                    rec.security_score,
                    rec.safety_rating,
                    rec.adaptability,
                    rec.antifragility,
                    rec.shannon_entropy,
                    rec.efficiency,
                    rec.flexibility,
                    rec.gpu_usage,
                    rec.projected_lucrativity,
                    rec.profitability,
                    rec.patch_complexity,
                    rec.patch_entropy,
                    rec.energy_consumption,
                    rec.resilience,
                    rec.network_latency,
                    rec.throughput,
                    rec.risk_index,
                    rec.maintainability,
                    rec.code_quality,
                    rec.ts,
                    menace_id,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def log_embedding_metrics(
        self,
        record_id: str,
        tokens: int,
        wall_time: float,
        index_latency: float,
        *,
        source: str = "",
    ) -> None:
        """Store embedding instrumentation details."""

        with self._connect() as conn:
            conn.execute(
            """
            INSERT INTO embedding_metrics(
                record_id, tokens, wall_time, index_latency, source, ts
            ) VALUES(?,?,?,?,?,?)
            """,
                (
                    record_id,
                    int(tokens),
                    float(wall_time),
                    float(index_latency),
                    source,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def log_embedding_staleness(
        self, origin_db: str, record_id: str, stale_seconds: float
    ) -> None:
        """Record staleness cost for an accessed embedding."""

        with self._connect() as conn:
            conn.execute(
            """
            INSERT INTO embedding_staleness(
                origin_db, record_id, stale_seconds, ts
            ) VALUES(?,?,?,?)
            """,
                (
                    origin_db,
                    record_id,
                    float(stale_seconds),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def log_training_stat(self, source: str, success: bool) -> None:
        """Record a training event for monitoring."""

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO training_stats(source, success, ts) VALUES(?,?,?)",
                (source, 1 if success else 0, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def log_eval(
        self,
        cycle: str,
        metric: str,
        value: float,
        *,
        source_menace_id: object | None = None,
    ) -> None:
        with self._connect() as conn:
            menace_id = source_menace_id or self.router.menace_id
            conn.execute(
                "INSERT INTO eval_metrics(cycle, metric, value, ts, source_menace_id) VALUES(?,?,?,?,?)",
                (cycle, metric, value, datetime.utcnow().isoformat(), menace_id),
            )
            conn.commit()

    def log_retrieval_metrics(
        self,
        origin_db: str,
        record_id: str,
        rank: int,
        hit: bool,
        tokens: int,
        score: float,
    ) -> None:
        """Persist per-result retrieval metrics."""

        with self._connect() as conn:
            conn.execute(
            """
            INSERT INTO retrieval_metrics(
                origin_db, record_id, rank, hit, tokens, score, ts
            ) VALUES(?,?,?,?,?,?,?)
            """,
                (
                    origin_db,
                    record_id,
                    int(rank),
                    1 if hit else 0,
                    int(tokens),
                    float(score),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def log_embedding_stat(
        self,
        db_name: str,
        tokens: int,
        wall_ms: float,
        store_ms: float,
        *,
        patch_id: str = "",
        db_source: str = "",
    ) -> None:
        """Record a single embedding statistics entry."""

        with self._connect() as conn:
            conn.execute(
            """
            INSERT INTO embedding_stats(
                db_name, tokens, wall_ms, store_ms, patch_id, db_source, ts
            ) VALUES(?,?,?,?,?,?,?)
            """,
                (
                    db_name,
                    int(tokens),
                    float(wall_ms),
                    float(store_ms),
                    patch_id,
                    db_source,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def log_retrieval_stat(
        self,
        session_id: str,
        origin_db: str,
        record_id: str,
        rank: int,
        hit: bool,
        hit_rate: float,
        tokens_injected: int,
        contribution: float,
        *,
        patch_id: str = "",
        db_source: str = "",
    ) -> None:
        """Persist retrieval statistics compatible with aggregation."""

        with self._connect() as conn:
            conn.execute(
            """
            INSERT INTO retrieval_stats(
                session_id, origin_db, record_id, rank, hit, hit_rate,
                tokens_injected, contribution, patch_id, db_source, ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    session_id,
                    origin_db,
                    record_id,
                    int(rank),
                    1 if hit else 0,
                    float(hit_rate),
                    int(tokens_injected),
                    float(contribution),
                    patch_id,
                    db_source,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def log_retriever_kpi(
        self,
        origin_db: str,
        win_rate: float,
        regret_rate: float,
        stale_cost: float,
        roi: float = 0.0,
        sample_count: float = 0.0,
    ) -> None:
        """Store aggregate KPIs for retrieval performance."""

        with self._connect() as conn:
            conn.execute(
            """
            INSERT INTO retriever_kpi(origin_db, win_rate, regret_rate, stale_penalty, sample_count, roi, ts)
            VALUES(?,?,?,?,?,?,?)
            """,
                (
                    origin_db,
                    float(win_rate),
                    float(regret_rate),
                    float(stale_cost),
                    float(sample_count),
                    float(roi),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def latest_retriever_kpi(self) -> Dict[str, Dict[str, float]]:
        """Return latest KPI metrics for each origin database.

        The most recent ``win_rate``, ``regret_rate``, ``stale_cost`` and
        ``sample_count`` values are retrieved from the ``retriever_kpi`` table.
        Results are returned as a mapping of ``origin_db`` to these metrics.
        When no KPI data has been recorded an empty mapping is returned.
        """

        with self._connect() as conn:
            cur = conn.execute(
            """
                SELECT origin_db, win_rate, regret_rate, stale_penalty, sample_count
                FROM (
                    SELECT origin_db, win_rate, regret_rate, stale_penalty, sample_count, ts
                    FROM retriever_kpi
                    ORDER BY ts DESC
                )
                GROUP BY origin_db
            """
            )
            rows = cur.fetchall()

        metrics: Dict[str, Dict[str, float]] = {}
        for origin, win, regret, stale, samples in rows:
            metrics[origin] = {
                "win_rate": float(win),
                "regret_rate": float(regret),
                "stale_cost": float(stale),
                "sample_count": float(samples),
            }
        return metrics

    def log_patch_outcome(
        self,
        patch_id: str,
        success: bool,
        vectors: Iterable[tuple[str, str]] | None = None,
        *,
        session_id: str = "",
        reverted: bool = False,
    ) -> None:
        """Record the outcome of a patch deployment and associated vectors."""

        entries = list(vectors or [])
        label = "positive" if success and not reverted else "negative"
        with self._connect() as conn:
            if entries:
                for origin_db, vec_id in entries:
                    conn.execute(
                        """
                    INSERT INTO patch_outcomes(patch_id, session_id, origin_db, vector_id, success, reverted, label, ts)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                        (
                            patch_id,
                            session_id,
                            origin_db,
                            vec_id,
                            1 if success else 0,
                            1 if reverted else 0,
                            label,
                            datetime.utcnow().isoformat(),
                        ),
                    )
                    conn.execute(
                        """
                    INSERT INTO retriever_stats(origin_db, wins, regrets)
                    VALUES(?,?,?)
                    ON CONFLICT(origin_db) DO UPDATE SET
                        wins = wins + excluded.wins,
                        regrets = regrets + excluded.regrets
                    """,
                        (origin_db, 1 if success else 0, 0 if success else 1),
                    )
            else:
                conn.execute(
                    """
                INSERT INTO patch_outcomes(patch_id, session_id, origin_db, vector_id, success, reverted, label, ts)
                VALUES(?,?,?,?,?,?,?,?)
            """,
                    (
                        patch_id,
                        session_id,
                        None,
                        None,
                        1 if success else 0,
                        1 if reverted else 0,
                        label,
                        datetime.utcnow().isoformat(),
                    ),
                )
            conn.commit()
        if _VEC_METRICS is not None and entries:
            try:
                _VEC_METRICS.update_outcome(
                    session_id,
                    list(entries),
                    contribution=0.0,
                    patch_id=patch_id,
                    win=bool(success and not reverted),
                    regret=bool((not success) or reverted),
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to update vector metrics outcome")

    def fetch_eval(
        self,
        cycle: str | None = None,
        *,
        source_menace_id: object | None = None,
        scope: Scope | str = Scope.LOCAL,
    ) -> List[tuple]:
        query = "SELECT cycle, metric, value, ts FROM eval_metrics"
        params: List[object] = []
        menace_id = source_menace_id or self.router.menace_id
        clause, scope_params = build_scope_clause("eval_metrics", scope, menace_id)
        query = apply_scope(query, clause)
        params.extend(scope_params)
        if cycle:
            query = apply_scope(query, "cycle=?")
            params.append(cycle)
        with self._connect() as conn:
            cur = conn.execute(query, params)
            return cur.fetchall()

    def fetch(
        self,
        limit: int | None = 100,
        *,
        start: str | None = None,
        end: str | None = None,
        source_menace_id: object | None = None,
        scope: Scope | str = Scope.LOCAL,
    ) -> object:
        """Return metrics optionally filtered by timestamp range and scope.

        Falls back to a list of dictionaries when pandas is unavailable.
        """
        query = (
            "SELECT bot, cpu, memory, response_time, disk_io, net_io, errors,"
            " revenue, expense, security_score, safety_rating, adaptability,"
            " antifragility, shannon_entropy, efficiency, flexibility, gpu_usage,"
            " projected_lucrativity, profitability, patch_complexity, patch_entropy,"
            " energy_consumption, resilience, network_latency, throughput,"
            " risk_index, maintainability, code_quality, ts FROM metrics"
        )
        params: List[object] = []
        clauses: List[str] = []
        menace_id = source_menace_id or self.router.menace_id
        scope_clause, scope_params = build_scope_clause(
            "metrics", Scope(scope), menace_id
        )
        if scope_clause:
            clauses.append(scope_clause)
            params.extend(scope_params)
        if start:
            clauses.append("ts >= ?")
            params.append(start)
        if end:
            clauses.append("ts <= ?")
            params.append(end)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        if pd is None:
            with self._connect() as conn:
                cur = conn.execute(query, params)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                return rows
        with self._connect() as conn:
            return pd.read_sql(query, conn, params=params)


@self_coding_managed
class DataBot:
    """Collect metrics, expose them to Prometheus and detect anomalies.

    A :class:`BaselineTracker` maintains rolling averages for ROI, error rate
    and test failures.  ``baseline_window`` controls the number of recent
    observations considered and ``anomaly_sensitivity`` adjusts how many
    standard deviations from the mean constitute a breach.
    """

    def __init__(
        self,
        db: MetricsDB | None = None,
        registry: CollectorRegistry | None = None,
        *,
        capital_bot: "CapitalManagementBot" | None = None,
        patch_db: PatchHistoryDB | None = None,
        start_server: bool | None = None,
        event_bus: UnifiedEventBus | None = None,
        evolution_db: EvolutionHistoryDB | None = None,
        settings: SandboxSettings | None = None,
        roi_drop_threshold: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        degradation_callback: Callable[[dict], None] | None = None,
        baseline_window: int | None = None,
        anomaly_sensitivity: float | None = None,
        smoothing_factor: float | None = None,
        trend_predictor: "TrendPredictor" | None = None,
        threshold_update_interval: float | None = None,
    ) -> None:
        self.db = db or MetricsDB()
        self.capital_bot = capital_bot
        self.patch_db = patch_db
        self.event_bus = event_bus
        self.evolution_db = evolution_db
        self.settings = settings or SandboxSettings()
        self.roi_drop_threshold = roi_drop_threshold
        self.error_threshold = error_threshold
        self.degradation_callback = degradation_callback
        self.test_failure_threshold = test_failure_threshold
        self._degradation_callbacks: list[Callable[[dict], None]] = []
        if degradation_callback:
            self._degradation_callbacks.append(degradation_callback)
        self.logger = logger
        self._current_cycle_id: int | None = None
        self.gauges: Dict[str, Gauge] = {}
        self.baseline_window = (
            baseline_window
            if baseline_window is not None
            else getattr(self.settings, "baseline_window", 10)
        )
        self.anomaly_sensitivity = (
            anomaly_sensitivity
            if anomaly_sensitivity is not None
            else getattr(self.settings, "anomaly_sensitivity", 1.0)
        )
        self.smoothing_factor = (
            smoothing_factor
            if smoothing_factor is not None
            else getattr(self.settings, "smoothing_factor", 0.1)
        )
        self.trend_predictor = trend_predictor
        if self.trend_predictor is None:
            try:  # pragma: no cover - optional dependency
                from .trend_predictor import TrendPredictor as _TP

                self.trend_predictor = _TP()
            except Exception:
                self.trend_predictor = None
        self._baseline: Dict[str, BaselineTracker] = {}
        self._ema_baseline: Dict[str, Dict[str, float]] = {}
        # Per-bot forecast history for adaptive thresholding.
        self._forecast_history: Dict[str, Dict[str, list[float]]] = {}
        # Cache per-bot thresholds to avoid unnecessary disk reads.  The
        # :meth:`reload_thresholds` method refreshes this mapping at runtime
        # when configuration files change.
        self._thresholds: Dict[str, ROIThresholds] = {}
        self.threshold_update_interval = (
            threshold_update_interval
            if threshold_update_interval is not None
            else getattr(self.settings, "threshold_update_interval", 60.0)
        )
        self._last_threshold_refresh: Dict[str, float] = {}
        if Gauge:
            self.registry = registry or CollectorRegistry()
            self.gauges = {
                "cpu": Gauge("bot_cpu", "CPU usage", ["bot"], registry=self.registry),
                "memory": Gauge(
                    "bot_memory", "Memory usage", ["bot"], registry=self.registry
                ),
                "response_time": Gauge(
                    "bot_response_time",
                    "Response time",
                    ["bot"],
                    registry=self.registry,
                ),
                "disk_io": Gauge(
                    "bot_disk_io", "Disk IO", ["bot"], registry=self.registry
                ),
                "net_io": Gauge(
                    "bot_net_io", "Network IO", ["bot"], registry=self.registry
                ),
                "errors": Gauge(
                    "bot_errors", "Error count", ["bot"], registry=self.registry
                ),
                "revenue": Gauge(
                    "bot_revenue", "Revenue", ["bot"], registry=self.registry
                ),
                "expense": Gauge(
                    "bot_expense", "Expense", ["bot"], registry=self.registry
                ),
                "security_score": Gauge(
                    "bot_security_score", "Security score", ["bot"], registry=self.registry
                ),
                "safety_rating": Gauge(
                    "bot_safety_rating", "Safety rating", ["bot"], registry=self.registry
                ),
                "adaptability": Gauge(
                    "bot_adaptability", "Adaptability", ["bot"], registry=self.registry
                ),
                "antifragility": Gauge(
                    "bot_antifragility", "Antifragility", ["bot"], registry=self.registry
                ),
                "shannon_entropy": Gauge(
                    "bot_shannon_entropy", "Shannon entropy", ["bot"], registry=self.registry
                ),
                "efficiency": Gauge(
                    "bot_efficiency", "Efficiency", ["bot"], registry=self.registry
                ),
                "flexibility": Gauge(
                    "bot_flexibility", "Flexibility", ["bot"], registry=self.registry
                ),
                "gpu_usage": Gauge(
                    "bot_gpu_usage", "GPU usage", ["bot"], registry=self.registry
                ),
                "projected_lucrativity": Gauge(
                    "bot_projected_lucrativity",
                    "Projected lucrativity",
                    ["bot"],
                    registry=self.registry,
                ),
                "profitability": Gauge(
                    "bot_profitability",
                    "Profitability",
                    ["bot"],
                    registry=self.registry,
                ),
                "patch_complexity": Gauge(
                    "bot_patch_complexity",
                    "Patch complexity",
                    ["bot"],
                    registry=self.registry,
                ),
                "patch_entropy": Gauge(
                    "bot_patch_entropy",
                    "Patch entropy",
                    ["bot"],
                    registry=self.registry,
                ),
                "energy_consumption": Gauge(
                    "bot_energy_consumption",
                    "Energy consumption",
                    ["bot"],
                    registry=self.registry,
                ),
                "resilience": Gauge(
                    "bot_resilience",
                    "Resilience",
                    ["bot"],
                    registry=self.registry,
                ),
                "network_latency": Gauge(
                    "bot_network_latency",
                    "Network latency",
                    ["bot"],
                    registry=self.registry,
                ),
                "throughput": Gauge(
                    "bot_throughput",
                    "Throughput",
                    ["bot"],
                    registry=self.registry,
                ),
                "risk_index": Gauge(
                    "bot_risk_index",
                    "Risk index",
                    ["bot"],
                    registry=self.registry,
                ),
                "maintainability": Gauge(
                    "bot_maintainability",
                    "Maintainability",
                    ["bot"],
                    registry=self.registry,
                ),
                "code_quality": Gauge(
                    "bot_code_quality",
                    "Code quality",
                    ["bot"],
                    registry=self.registry,
                ),
            }
            if start_server or os.getenv("METRICS_PORT"):
                from .metrics_exporter import start_metrics_server

                port = int(os.getenv("METRICS_PORT", "8001"))
                start_metrics_server(port)

    def subscribe_threshold_breaches(
        self, callback: Callable[[dict], None]
    ) -> None:
        """Subscribe *callback* to threshold breach events."""
        if not self.event_bus:
            raise RuntimeError("event bus not configured")
        self.event_bus.subscribe(
            "data:threshold_breach", lambda _t, e: callback(e)
        )

    # ------------------------------------------------------------------
    def subscribe_degradation(self, callback: Callable[[dict], None]) -> None:
        """Register *callback* for bot degradation notifications.

        Events published under ``bot:degraded`` include ROI, error and test
        failure deltas via the ``delta_roi``, ``delta_errors`` and
        ``delta_tests_failed`` keys.
        """
        if self.event_bus:
            self.event_bus.subscribe("bot:degraded", lambda _t, e: callback(e))
            try:
                cb_name = getattr(
                    callback, "__qualname__", getattr(callback, "__name__", str(callback))
                )
                logger.info("degradation monitoring enabled via %s", cb_name)
                self.event_bus.publish("bot:registered", {"callback": cb_name})
            except Exception:
                logger.exception("failed to emit bot:registered event")
        else:
            self._degradation_callbacks.append(callback)
            self.degradation_callback = callback

    def start_monitoring(self, interval: float) -> threading.Thread:
        """Periodically invoke :meth:`check_degradation` for known bots.

        Bots are discovered from the existing baseline trackers which are
        seeded via :meth:`check_degradation` when a bot registers.  The
        monitoring loop runs in a background daemon thread and publishes a
        ``data:monitoring_started`` event on the :class:`UnifiedEventBus`
        when available.
        """

        def _monitor() -> None:
            while True:
                bots = list(self._baseline.keys())
                for bot in bots:
                    try:
                        with self.db._connect() as conn:  # type: ignore[attr-defined]
                            row = conn.execute(
                                "SELECT revenue, expense, errors, tests_failed "
                                "FROM metrics WHERE bot=? "
                                "ORDER BY id DESC LIMIT 1",
                                (bot,),
                            ).fetchone()
                        if not row:
                            continue
                        revenue, expense, errors, tests_failed = row
                        roi = float(revenue) - float(expense)
                        self.check_degradation(
                            bot, roi, float(errors), float(tests_failed)
                        )
                    except Exception as exc:
                        self.logger.exception(
                            "monitoring loop failed for %s: %s", bot, exc
                        )
                        if self.event_bus:
                            try:
                                self.event_bus.publish(
                                    "data:monitoring_error",
                                    {"bot": bot, "error": str(exc)},
                                )
                            except Exception:
                                self.logger.exception(
                                    "failed to publish monitoring error event"
                                )
                time.sleep(interval)

        thread = threading.Thread(
            target=_monitor, name="data-bot-monitor", daemon=True
        )
        thread.start()
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "data:monitoring_started", {"interval": interval}
                )
            except Exception:
                self.logger.exception(
                    "failed to publish monitoring started event"
                )
        return thread

    def collect(
        self,
        bot: str,
        response_time: float = 0.0,
        errors: int = 0,
        tests_failed: int = 0,
        tests_run: int = 0,
        revenue: float = 0.0,
        expense: float = 0.0,
        *,
        security_score: float = 0.0,
        safety_rating: float = 0.0,
        adaptability: float = 0.0,
        antifragility: float = 0.0,
        shannon_entropy: float = 0.0,
        efficiency: float | None = None,
        flexibility: float = 0.0,
        gpu_usage: float = 0.0,
        projected_lucrativity: float = 0.0,
        profitability: float | None = None,
        patch_complexity: float = 0.0,
        patch_entropy: float = 0.0,
        energy_consumption: float | None = None,
        resilience: float = 0.0,
        network_latency: float = 0.0,
        throughput: float = 0.0,
        risk_index: float = 0.0,
        maintainability: float = 0.0,
        code_quality: float = 0.0,
        bottleneck: float | None = None,
    ) -> MetricRecord:
        if psutil:
            io = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            disk = float(io.read_bytes + io.write_bytes)
            netio = float(net.bytes_sent + net.bytes_recv)
        else:  # fallback using standard library stats
            cpu = 0.0
            try:
                if hasattr(os, "getloadavg") and os.cpu_count():
                    load = os.getloadavg()[0]
                    cpu = min(100.0, 100.0 * load / (os.cpu_count() or 1))
            except Exception:
                cpu = 0.0
            try:
                import resource

                mem = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (
                    1024 * 1024
                )
            except Exception:
                mem = 0.0
            try:
                import shutil

                disk = float(shutil.disk_usage("/").used)
            except Exception:
                disk = 0.0
            try:
                with open("/proc/net/dev", "r", encoding="utf-8") as fh:
                    next(fh)
                    next(fh)
                    totals = [line.split() for line in fh]
                    netio = sum(
                        float(cols[1]) + float(cols[9])
                        for cols in totals
                        if len(cols) >= 10
                    )
            except Exception:
                netio = 0.0
        if efficiency is None:
            efficiency = float(max(0.0, 100.0 - cpu))
        if energy_consumption is None:
            energy_consumption = float(cpu)
        if profitability is None:
            profitability = float(revenue - expense)
        if not network_latency:
            network_latency = float(netio) / 1000.0 if netio else 0.0
        if not throughput:
            throughput = 1000.0 / (network_latency + 1.0)
        if not risk_index:
            risk_index = max(0.0, 100.0 - (security_score + safety_rating) / 2.0)
        if self.patch_db and (not patch_complexity or not patch_entropy):
            try:
                with self.patch_db._connect() as conn:
                    comp_rows = conn.execute(
                        "SELECT complexity_after, entropy_after FROM patch_history ORDER BY id DESC LIMIT 5"
                    ).fetchall()
                if comp_rows:
                    if not patch_complexity:
                        patch_complexity = float(
                            sum(float(r[0] or 0.0) for r in comp_rows) / len(comp_rows)
                        )
                    if not patch_entropy:
                        patch_entropy = float(
                            sum(float(r[1] or 0.0) for r in comp_rows) / len(comp_rows)
                        )
            except Exception:
                if not patch_complexity:
                    patch_complexity = 0.0
                if not patch_entropy:
                    patch_entropy = 0.0
        if not resilience:
            resilience = 100.0 / float(errors + 1)
        rec = MetricRecord(
            bot=bot,
            cpu=cpu,
            memory=mem,
            response_time=response_time,
            disk_io=disk,
            net_io=netio,
            errors=errors,
            tests_failed=tests_failed,
            tests_run=tests_run,
            revenue=revenue,
            expense=expense,
            security_score=security_score,
            safety_rating=safety_rating,
            adaptability=adaptability,
            antifragility=antifragility,
            shannon_entropy=shannon_entropy,
            efficiency=efficiency,
            flexibility=flexibility,
            gpu_usage=gpu_usage,
            projected_lucrativity=projected_lucrativity,
            profitability=profitability,
            patch_complexity=patch_complexity,
            patch_entropy=patch_entropy,
            energy_consumption=energy_consumption,
            resilience=resilience,
            network_latency=network_latency,
            throughput=throughput,
            risk_index=risk_index,
            maintainability=maintainability,
            code_quality=code_quality,
        )
        self.db.add(rec)
        # compute efficiency/bottleneck metrics if not explicitly provided
        if bottleneck is None:
            bottleneck = float(errors)
        if self.evolution_db and self._current_cycle_id is not None:
            try:
                self.evolution_db.update_cycle(
                    self._current_cycle_id, efficiency, bottleneck
                )
            except Exception as exc:
                self.logger.exception("failed to update evolution cycle: %s", exc)
        current_roi = revenue - expense
        if self.event_bus:
            try:
                self.event_bus.publish("metrics:new", asdict(rec))
            except Exception as exc:
                self.logger.exception("failed to publish metrics event: %s", exc)
        if self.event_bus or self.degradation_callback or self._degradation_callbacks:
            self.check_degradation(
                bot,
                current_roi,
                float(errors),
                test_failures=float(tests_failed),
            )
        for name, gauge in self.gauges.items():
            gauge.labels(bot=bot).set(getattr(rec, name))
        if self.patch_db:
            try:
                rate = self.patch_db.success_rate()
                self.db.log_eval("system", "patch_success_rate", float(rate))
            except Exception as exc:
                self.logger.exception("failed to query patch DB: %s", exc)
        if self.capital_bot:
            try:
                energy = self.capital_bot.energy_score(
                    load=0.0,
                    success_rate=1.0,
                    deploy_eff=1.0,
                    failure_rate=errors,
                )
                self.db.log_eval("system", "avg_energy_score", float(energy))
            except Exception as exc:
                self.logger.exception("failed to query capital bot: %s", exc)
        return rec

    def forecast_metrics(
        self,
        bot: str,
        *,
        roi: float,
        errors: float,
        tests_failed: float = 0.0,
    ) -> Dict[str, tuple[float, float, float]]:
        """Forecast next-cycle metrics using ``TrendPredictor``.

        Returns a mapping of metric name to ``(prediction, low, high)`` tuples.
        Low/high bounds are derived from the current value when confidence
        intervals are unavailable.
        """

        prediction = None
        if self.trend_predictor is not None:
            try:
                self.trend_predictor.train()
                prediction = self.trend_predictor.predict_future_metrics()
            except Exception:  # pragma: no cover - prediction best effort
                prediction = None

        pred_roi = prediction.roi if prediction else roi
        pred_err = prediction.errors if prediction else errors
        pred_fail = tests_failed

        return {
            "roi": (pred_roi, min(pred_roi, roi), max(pred_roi, roi)),
            "errors": (pred_err, min(pred_err, errors), max(pred_err, errors)),
            "tests_failed": (
                pred_fail,
                min(pred_fail, tests_failed),
                max(pred_fail, tests_failed),
            ),
        }

    def check_degradation(
        self,
        bot: str,
        roi: float,
        errors: float,
        test_failures: float = 0.0,
        *,
        callback: Callable[[dict], None] | None = None,
    ) -> bool:
        """Compare current metrics against configured degradation thresholds.

        Metrics are evaluated against a rolling baseline maintained by
        :class:`BaselineTracker`.  If a degradation is detected the event is
        published on the event bus as ``bot:degraded`` or, when no bus is
        configured, a provided *callback* is invoked.  Returns ``True`` when the
        bot breached either threshold.
        """

        if bot not in self._thresholds:
            try:
                self.reload_thresholds(bot)
            except Exception:  # pragma: no cover - best effort
                pass
        tracker = self._baseline.setdefault(
            bot, BaselineTracker(window=self.baseline_window)
        )
        ema = self._ema_baseline.setdefault(
            bot,
            {"roi": float(roi), "errors": float(errors), "tests_failed": float(test_failures)},
        )
        avg_roi = float(ema["roi"])
        avg_err = float(ema["errors"])
        avg_fail = float(ema["tests_failed"])
        delta_roi = roi - avg_roi
        delta_err = errors - avg_err
        delta_fail = test_failures - avg_fail

        prediction = None
        if self.trend_predictor is not None:
            try:
                self.trend_predictor.train()
                prediction = self.trend_predictor.predict_future_metrics()
            except Exception:  # pragma: no cover - prediction best effort
                prediction = None

        now = time.time()
        thresholds = self._thresholds.get(bot)
        last = self._last_threshold_refresh.get(bot, 0.0)
        if (
            thresholds is None
            or now - last >= self.threshold_update_interval
        ):
            thresholds = adaptive_thresholds(
                bot,
                roi_baseline=avg_roi,
                error_baseline=avg_err,
                failure_baseline=avg_fail,
                prediction=prediction,
                predictor=self.trend_predictor,
            )
            self._thresholds[bot] = thresholds
            try:
                save_sc_thresholds(
                    bot,
                    roi_drop=thresholds.roi_drop,
                    error_increase=thresholds.error_threshold,
                    test_failure_increase=thresholds.test_failure_threshold,
                )
            except Exception:  # pragma: no cover - best effort
                pass
            self._last_threshold_refresh[bot] = now
        roi_thresh = thresholds.roi_drop
        err_thresh = thresholds.error_threshold
        fail_thresh = thresholds.test_failure_threshold

        pred_roi = prediction.roi if prediction else avg_roi
        pred_err = prediction.errors if prediction else avg_err
        pred_fail = avg_fail

        roi_low = min(pred_roi, avg_roi)
        roi_high = max(pred_roi, avg_roi)
        err_low = min(pred_err, avg_err)
        err_high = max(pred_err, avg_err)
        fail_low = min(pred_fail, avg_fail)
        fail_high = max(pred_fail, avg_fail)

        fhist = self._forecast_history.setdefault(
            bot, {"roi": [], "errors": [], "tests_failed": []}
        )
        fhist["roi"].append(pred_roi)
        fhist["errors"].append(pred_err)
        fhist["tests_failed"].append(pred_fail)

        event = {
            "bot": bot,
            "roi": roi,
            "errors": errors,
            "predicted_roi": pred_roi,
            "predicted_errors": pred_err,
            "predicted_tests_failed": pred_fail,
            "roi_confidence_low": roi_low,
            "roi_confidence_high": roi_high,
            "error_confidence_low": err_low,
            "error_confidence_high": err_high,
            "test_failure_confidence_low": fail_low,
            "test_failure_confidence_high": fail_high,
            "forecast_model": "trend",
            "forecast_confidence": 0.0,
            "delta_roi": delta_roi,
            "delta_errors": delta_err,
            "delta_tests_failed": delta_fail,
            "test_failures": test_failures,
            "roi_baseline": avg_roi,
            "errors_baseline": avg_err,
            "tests_failed_baseline": avg_fail,
            "roi_threshold": roi_thresh,
            "error_threshold": err_thresh,
            "test_failure_threshold": fail_thresh,
            "roi_breach": delta_roi <= roi_thresh,
            "error_breach": delta_err >= err_thresh,
            "test_failure_breach": delta_fail > fail_thresh,
        }
        # Provide a concise summary for consumers that only require high-level
        # degradation indicators.
        event.update(
            {
                "roi_drop": delta_roi,
                "error_rate": errors,
                "tests_failed": test_failures,
            }
        )
        self.logger.info("degradation metrics: %s", event)
        alpha = self.smoothing_factor
        ema["roi"] = alpha * roi + (1.0 - alpha) * avg_roi
        ema["errors"] = alpha * errors + (1.0 - alpha) * avg_err
        ema["tests_failed"] = alpha * test_failures + (1.0 - alpha) * avg_fail
        tracker.update(roi=roi, errors=errors, tests_failed=test_failures)
        degraded = (
            event["roi_breach"]
            or event["error_breach"]
            or event["test_failure_breach"]
        )

        callbacks = []
        if callback:
            callbacks.append(callback)
        callbacks.extend(self._degradation_callbacks)
        if self.event_bus:
            try:
                self.event_bus.publish("metrics:delta", event)
                if event["roi_breach"] or event["error_breach"]:
                    self.event_bus.publish("data:threshold_breach", event)
                if degraded:
                    self.event_bus.publish("bot:degraded", event)
            except Exception as exc:
                self.logger.exception("failed to publish metrics event: %s", exc)
        elif degraded and callbacks:
            for cb in callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    self.logger.exception(
                        "degradation callback failed: %s", exc
                    )
        return degraded

    def log_evolution_cycle(
        self,
        action: str,
        before: float,
        after: float,
        roi: float,
        predicted_roi: float = 0.0,
        *,
        patch_success: float | None = None,
        roi_delta: float | None = None,
        roi_trend: float | None = None,
        anomaly_count: float | None = None,
        efficiency: float | None = None,
        bottleneck: float | None = None,
        patch_id: int | None = None,
        workflow_id: int | None = None,
        trending_topic: str | None = None,
        reverted: bool | None = None,
        reason: str = "",
        trigger: str = "",
        parent_event_id: int | None = None,
    ) -> None:
        """Record an evolution event via the connected database and metrics DB."""
        if not self.evolution_db:
            return
        try:
            self._current_cycle_id = self.evolution_db.add(
                EvolutionEvent(
                    action=action,
                    before_metric=before,
                    after_metric=after,
                    roi=roi,
                    predicted_roi=predicted_roi,
                    efficiency=efficiency or 0.0,
                    bottleneck=bottleneck or 0.0,
                    patch_id=patch_id,
                    workflow_id=workflow_id,
                    trending_topic=trending_topic,
                    reason=reason,
                    trigger=trigger,
                    parent_event_id=parent_event_id,
                )
            )
        except Exception:
            logger.exception("failed to add evolution event")
        if patch_success is not None:
            try:
                self.db.log_eval(action, "patch_success_rate", float(patch_success))
            except Exception:
                logger.exception("failed to log patch_success_rate")
        if roi_delta is not None:
            try:
                self.db.log_eval(action, "roi_delta", float(roi_delta))
            except Exception:
                logger.exception("failed to log roi_delta")
        if roi_trend is not None:
            try:
                self.db.log_eval(action, "roi_trend", float(roi_trend))
            except Exception:
                logger.exception("failed to log roi_trend")
        if anomaly_count is not None:
            try:
                self.db.log_eval(action, "anomaly_count", float(anomaly_count))
            except Exception:
                logger.exception("failed to log anomaly_count")
        if trending_topic is not None:
            try:
                self.db.log_eval(action, "trending_topic", 1.0)
            except Exception:
                logger.exception("failed to log trending_topic")
        if reverted is not None:
            try:
                self.db.log_eval(action, "patch_reverted", float(reverted))
            except Exception:
                logger.exception("failed to log patch_reverted")

    def log_workflow_evolution(
        self,
        workflow_id: int,
        variant: str,
        baseline_roi: float,
        variant_roi: float,
        *,
        mutation_id: int | None = None,
    ) -> int | None:
        """Record workflow variant evaluation details."""

        if not self.evolution_db:
            return None
        roi_delta = variant_roi - baseline_roi
        try:
            event_id = self.evolution_db.log_workflow_evolution(
                workflow_id=workflow_id,
                variant=variant,
                baseline_roi=baseline_roi,
                variant_roi=variant_roi,
                roi_delta=roi_delta,
                mutation_id=mutation_id,
            )
        except Exception:
            logger.exception("failed to log workflow evolution")
            return None
        cycle = f"workflow:{workflow_id}:{variant}"
        try:
            self.db.log_eval(cycle, "baseline_roi", float(baseline_roi))
            self.db.log_eval(cycle, "variant_roi", float(variant_roi))
            self.db.log_eval(cycle, "roi_delta", float(roi_delta))
        except Exception:
            logger.exception("failed to log workflow evolution metrics")
        return event_id

    def average_errors(self, bot: str, limit: int = 10) -> float:
        """Return average error count for ``bot`` over recent records."""

        try:
            df = self.db.fetch(limit)
            if hasattr(df, "empty"):
                df = df[df["bot"] == bot]
                if df.empty:
                    return 0.0
                return float(df["errors"].mean())
            if isinstance(df, list):
                rows = [r for r in df if r.get("bot") == bot]
                if not rows:
                    return 0.0
                return float(sum(r.get("errors", 0.0) for r in rows) / len(rows))
        except Exception:
            return 0.0
        return 0.0

    def average_test_failures(self, bot: str, limit: int = 10) -> float:
        """Return average failed test count for ``bot`` over recent records."""

        try:
            df = self.db.fetch(limit)
            if hasattr(df, "empty"):
                df = df[df["bot"] == bot]
                if df.empty:
                    return 0.0
                return float(df["tests_failed"].mean())
            if isinstance(df, list):
                rows = [r for r in df if r.get("bot") == bot]
                if not rows:
                    return 0.0
                return float(
                    sum(r.get("tests_failed", 0.0) for r in rows) / len(rows)
                )
        except Exception:
            return 0.0
        return 0.0

    def reload_thresholds(self, bot: str | None = None) -> ROIThresholds:
        """Reload thresholds for ``bot`` from configuration.

        This fetches fresh values via :func:`self_coding_thresholds.get_thresholds`
        and updates the local cache so long running processes can pick up
        runtime edits to ``config/self_coding_thresholds.yaml``.
        """

        t = load_sc_thresholds(bot, self.settings)
        roi_drop = (
            self.roi_drop_threshold
            if self.roi_drop_threshold is not None
            else t.roi_drop
        )
        error_thresh = (
            self.error_threshold
            if self.error_threshold is not None
            else t.error_increase
        )
        fail_thresh = (
            self.test_failure_threshold
            if self.test_failure_threshold is not None
            else t.test_failure_increase
        )
        rt = ROIThresholds(
            roi_drop=roi_drop,
            error_threshold=error_thresh,
            test_failure_threshold=fail_thresh,
        )
        key = bot or ""
        self._thresholds[key] = rt

        # Persist explicit overrides back to the configuration file so
        # subsequent processes observe the updated thresholds without a
        # restart.
        if bot and (
            self.roi_drop_threshold is not None
            or self.error_threshold is not None
            or self.test_failure_threshold is not None
        ):
            try:  # pragma: no cover - best effort persistence
                save_sc_thresholds(
                    bot,
                    roi_drop=roi_drop if self.roi_drop_threshold is not None else None,
                    error_increase=
                        error_thresh if self.error_threshold is not None else None,
                    test_failure_increase=
                        fail_thresh if self.test_failure_threshold is not None else None,
                )
            except Exception:
                self.logger.exception("failed to persist thresholds for %s", bot)
        return rt

    def get_thresholds(self, bot: str | None = None) -> ROIThresholds:
        """Return cached thresholds for ``bot``.

        When a value is not present, :meth:`reload_thresholds` is invoked to
        populate the cache.
        """

        key = bot or ""
        if key not in self._thresholds:
            return self.reload_thresholds(bot)
        return self._thresholds[key]

    def update_thresholds(
        self,
        bot: str,
        *,
        roi_drop: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        forecast: Dict[str, float] | None = None,
    ) -> None:
        """Persist new thresholds for ``bot`` and optionally notify API."""
        save_sc_thresholds(
            bot,
            roi_drop=roi_drop,
            error_increase=error_threshold,
            test_failure_increase=test_failure_threshold,
        )
        if forecast is not None:
            hist = self._forecast_history.setdefault(
                bot, {"roi": [], "errors": [], "tests_failed": []}
            )
            for k, v in forecast.items():
                hist.setdefault(k, []).append(float(v))

        api_url = os.getenv("SELF_CODING_THRESHOLD_API")
        if api_url:
            payload = {
                "bot": bot,
                "roi_drop": roi_drop,
                "error_threshold": error_threshold,
                "test_failure_threshold": test_failure_threshold,
            }
            try:
                import requests

                requests.post(api_url, json=payload, timeout=5)
            except Exception:
                self.logger.exception("failed to update thresholds via API")

    def roi(self, bot: str) -> float:
        """Return ROI for a bot via the capital manager if available."""
        if not self.capital_bot:
            return 0.0
        try:
            return float(self.capital_bot.bot_roi(bot))
        except Exception:
            return 0.0

    def worst_bot(self, metric: str = "errors", limit: int = 100) -> str | None:
        """Return the bot with the worst average ``metric``."""
        try:
            df = self.db.fetch(limit)
            if hasattr(df, "empty"):
                if getattr(df, "empty", True) or metric not in df.columns:
                    return None
                grp = df.groupby("bot")[metric].mean().sort_values(ascending=False)
                if grp.empty:
                    return None
                return str(grp.index[0])
            if isinstance(df, list) and df:
                totals: dict[str, float] = {}
                counts: dict[str, int] = {}
                for row in df:
                    b = row.get("bot")
                    val = float(row.get(metric, 0.0))
                    totals[b] = totals.get(b, 0.0) + val
                    counts[b] = counts.get(b, 0) + 1
                if not totals:
                    return None
                avg = {k: totals[k] / counts.get(k, 1) for k in totals}
                return max(avg, key=avg.get)
        except Exception:
            return None
        return None

    def engagement_delta(self, limit: int = 50) -> float:
        """Return revenue minus expense normalised by expense."""
        df = self.db.fetch(limit)
        if hasattr(df, "empty"):
            if getattr(df, "empty", True):
                return 0.0
            revenue = float(df["revenue"].sum())
            expense = float(df["expense"].sum()) or 1.0
        else:
            if not isinstance(df, list) or not df:
                return 0.0
            revenue = sum(r.get("revenue", 0.0) for r in df)
            expense = sum(r.get("expense", 0.0) for r in df) or 1.0
        return (revenue - expense) / expense

    def long_term_roi_trend(self, limit: int = 200) -> float:
        """Return the ROI change between early and late metrics."""
        df = self.db.fetch(limit)
        if hasattr(df, "empty"):
            if getattr(df, "empty", True):
                return 0.0
            df["roi"] = df["revenue"] - df["expense"]
            half = len(df) // 2 or 1
            first = float(df.iloc[:half]["roi"].mean())
            last = float(df.iloc[half:]["roi"].mean())
        else:
            if not isinstance(df, list) or not df:
                return 0.0
            rois = [float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df]
            half = len(rois) // 2 or 1
            first = sum(rois[:half]) / half
            last = sum(rois[half:]) / max(len(rois) - half, 1)
        if first == 0.0:
            return 0.0
        return (last - first) / abs(first)

    def forecast_roi_drop(self, limit: int = 100) -> float:
        """Predict ROI change for the next cycle using linear regression."""
        df = self.db.fetch(limit)
        if hasattr(df, "empty"):
            if getattr(df, "empty", True):
                return 0.0
            df["roi"] = df["revenue"] - df["expense"]
            y = df["roi"].tolist()
        else:
            if not isinstance(df, list) or not df:
                return 0.0
            y = [float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df]
        if len(y) < 2:
            return 0.0
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np

            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, np.array(y))
            pred = float(model.predict([[len(y)]])[0])
            return pred - y[-1]
        except Exception:
            return y[-1] - y[-2]

    def record_metrics(
        self,
        bot: str,
        roi: float,
        errors: float,
        *,
        tests_failed: float = 0.0,
    ) -> None:
        """Persist basic ROI and error metrics for ``bot``."""
        try:
            rec = MetricRecord(
                bot=bot,
                cpu=0.0,
                memory=0.0,
                response_time=0.0,
                disk_io=0.0,
                net_io=0.0,
                errors=int(errors),
                tests_failed=int(tests_failed),
                revenue=float(roi),
                expense=0.0,
            )
            self.db.add(rec)
        except Exception:
            self.logger.exception("failed to record metrics for %s", bot)

    def record_validation(
        self, bot: str, module: str, passed: bool, flags: List[str] | None = None
    ) -> None:
        """Persist patch validation results for later analysis."""
        try:
            with self.db._connect() as conn:  # type: ignore[attr-defined]
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS patch_validation(" "bot TEXT, module TEXT, passed INTEGER, flags TEXT, ts TEXT)"
                )
                conn.execute(
                    "INSERT INTO patch_validation(bot,module,passed,flags,ts) VALUES(?,?,?,?,?)",
                    (
                        bot,
                        module,
                        1 if passed else 0,
                        json.dumps(flags or []),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
        except Exception:
            self.logger.exception("failed to record patch validation")

    def record_test_failure(self, bot: str, count: int) -> None:
        """Log recent failing test count for ``bot`` and check degradation."""
        try:
            self.db.log_eval(bot, "test_failures", float(count))
        except Exception:
            self.logger.exception("failed to log test failures")
        roi = 0.0
        errors = 0.0
        try:
            roi = self.roi(bot)
        except Exception:
            self.logger.exception("failed to fetch ROI for %s", bot)
        try:
            errors = self.average_errors(bot)
        except Exception:
            self.logger.exception("failed to fetch error averages for %s", bot)
        try:
            self.check_degradation(
                bot,
                float(roi),
                float(errors),
                test_failures=float(count),
            )
        except Exception:
            self.logger.exception("degradation check failed for test failures")

    @staticmethod
    def detect_anomalies(
        df: object,
        field: str,
        threshold: float = 3.0,
        *,
        metrics_db: MetricsDB | None = None,
    ) -> List[int]:
        """Return indices of rows considered anomalies.

        When PyTorch or scikit-learn are available this delegates to
        :mod:`menace.anomaly_detection` and logs the produced anomaly scores
        via ``metrics_db``.  It falls back to a simple standard deviation based
        approach otherwise.
        """
        values: List[float]
        if pd is not None and hasattr(df, "empty") and not getattr(df, "empty", True):
            if field not in df.columns:
                return []
            values = df[field].tolist()
        elif isinstance(df, list):  # pragma: no cover - fallback when pandas missing
            values = [float(row.get(field, 0)) for row in df]
        else:
            return []

        if not values:
            return []

        try:
            from . import anomaly_detection

            scores = anomaly_detection.anomaly_scores(values)
            if metrics_db:
                for score in scores:
                    metrics_db.log_eval("anomaly", field, float(score))
            mean_s = sum(scores) / len(scores)
            std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0
            return [i for i, s in enumerate(scores) if s > mean_s + threshold * std_s]
        except Exception:
            logger.exception(
                "anomaly detection failed, falling back to standard deviation"
            )

        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
        return [i for i, v in enumerate(values) if v > mean + threshold * std]

    @staticmethod
    def complexity_score(df: object) -> float:
        """Return a rough complexity score for metrics data."""
        if pd is None:
            if not isinstance(df, list) or not df:
                return 0.0
            cpu = sum(r.get("cpu", 0.0) for r in df) / len(df)
            mem = sum(r.get("memory", 0.0) for r in df) / len(df)
            return float(cpu + mem)
        if getattr(df, "empty", True):
            return 0.0
        return float(df["cpu"].mean() + df["memory"].mean())


__all__ = ["MetricRecord", "MetricsDB", "DataBot"]
