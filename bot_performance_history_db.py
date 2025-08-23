from __future__ import annotations

"""Store per-bot performance metrics for RL training."""

from dataclasses import dataclass, fields, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Sequence
import asyncio
import math
import os
import logging
import sqlite3
import threading
import json
import time

from .env_config import BOT_PERFORMANCE_DB
from db_router import GLOBAL_ROUTER, init_db_router

logger = logging.getLogger(__name__)
router = GLOBAL_ROUTER or init_db_router("bot_performance_history")

DEFAULT_FIELD_INFO: dict[str, str] = {
    "bot": "TEXT",
    "cpu": "REAL",
    "memory": "REAL",
    "response_time": "REAL",
    "errors": "INTEGER",
    "roi": "REAL",
    "score": "REAL",
    "disk_io": "REAL DEFAULT 0",
    "net_io": "REAL DEFAULT 0",
    "revenue": "REAL DEFAULT 0",
    "expense": "REAL DEFAULT 0",
    "metadata": "TEXT DEFAULT '{}'",
    "ts": "TEXT",
}

@dataclass
class PerformanceRecord:
    bot: str
    cpu: float
    memory: float
    response_time: float
    errors: int
    roi: float
    score: float
    disk_io: float = 0.0
    net_io: float = 0.0
    revenue: float = 0.0
    expense: float = 0.0
    metadata: str = "{}"
    ts: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        numeric_fields = [
            self.cpu,
            self.memory,
            self.response_time,
            self.errors,
            self.roi,
            self.score,
            self.disk_io,
            self.net_io,
            self.revenue,
            self.expense,
        ]
        if any(not math.isfinite(v) for v in numeric_fields):
            raise ValueError("non-finite value provided for performance metric")
        if any(v < 0 for v in [self.cpu, self.memory, self.response_time, self.errors]):
            raise ValueError("metrics must be non-negative")
        try:
            json.loads(self.metadata)
        except Exception as exc:
            raise ValueError("metadata must be JSON") from exc
        if isinstance(self.ts, str):
            self.ts = datetime.fromisoformat(self.ts)


@dataclass
class AddResult:
    """Result of inserting a record into the history DB."""

    rowid: int
    record: PerformanceRecord


class BotPerformanceHistoryDB:
    """SQLite-backed log of performance assessments."""

    def __init__(
        self,
        path: Path | str | None = None,
        field_info: dict[str, str] | None = None,
        *,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> None:
        db_path = Path(path or os.environ.get("BOT_PERFORMANCE_DB", BOT_PERFORMANCE_DB))
        logger.info("using performance history db at %s", db_path.resolve())
        self.path = str(db_path)
        self._lock = threading.Lock()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        def _conn() -> sqlite3.Connection:
            return router.get_connection("bot_performance")

        field_info = field_info or DEFAULT_FIELD_INFO

        cols_def = ",".join(f"{k} {v}" for k, v in field_info.items())
        with _conn() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS bot_performance ({cols_def})")

            existing = [r[1] for r in conn.execute("PRAGMA table_info(bot_performance)").fetchall()]
            for col, col_type in field_info.items():
                if col not in existing:
                    try:
                        conn.execute(
                            f"ALTER TABLE bot_performance ADD COLUMN {col} {col_type}"
                        )
                    except Exception:
                        logger.exception("failed adding column %s", col)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bot_ts ON bot_performance(bot, ts DESC)"
            )
            conn.commit()

        self._connect = _conn

    def _execute(
        self, query: str, params: Sequence[object] | None = None, *, commit: bool = False
    ) -> sqlite3.Cursor:
        params = params or []
        for i in range(self.max_retries):
            try:
                with self._lock, self._connect() as conn:
                    cur = conn.execute(query, params)
                    if commit:
                        conn.commit()
                    return cur
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and i < self.max_retries - 1:
                    logger.warning("db locked, retry %s/%s", i + 1, self.max_retries)
                    time.sleep(self.retry_delay * (2 ** i))
                    continue
                raise

    # --------------------------------------------------------------
    def __enter__(self) -> "BotPerformanceHistoryDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            logger.exception("failed to close bot performance db during cleanup")

    def close(self) -> None:
        pass

    def add(self, rec: PerformanceRecord) -> AddResult:
        data = {f.name: getattr(rec, f.name) for f in fields(PerformanceRecord)}
        data["ts"] = rec.ts.isoformat()
        cols = ",".join(data.keys())
        placeholders = ",".join(["?"] * len(data))
        try:
            cur = self._execute(
                f"INSERT INTO bot_performance({cols}) VALUES({placeholders})",
                list(data.values()),
                commit=True,
            )
            return AddResult(rowid=int(cur.lastrowid), record=rec)
        except sqlite3.Error as exc:
            logger.exception("failed inserting performance record: %s", exc)
            raise

    async def add_async(self, rec: PerformanceRecord) -> AddResult:
        """Asynchronous wrapper around :meth:`add`."""
        return await asyncio.to_thread(self.add, rec)

    def history(
        self, bot: str, limit: int = 50
    ) -> List[Tuple[float, float, float, int, float, float, float, float, float, float, str, str]]:
        try:
            cur = self._execute(
                "SELECT cpu,memory,response_time,errors,roi,score,disk_io,net_io,revenue,expense,metadata,ts FROM bot_performance WHERE bot=? ORDER BY ts DESC LIMIT ?",
                (bot, limit),
            )
            return cur.fetchall()
        except sqlite3.Error as exc:
            logger.exception("failed fetching history for %s: %s", bot, exc)
            return []

    async def history_async(
        self, bot: str, limit: int = 50
    ) -> List[Tuple[float, float, float, int, float, float, float, float, float, float, str, str]]:
        """Asynchronous wrapper around :meth:`history`."""
        return await asyncio.to_thread(self.history, bot, limit)

    def prune_old_entries(self, max_age_days: int) -> int:
        """Delete records older than ``max_age_days``."""
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        cur = self._execute(
            "DELETE FROM bot_performance WHERE ts < ?",
            (cutoff,),
            commit=True,
        )
        return cur.rowcount


__all__ = [
    "PerformanceRecord",
    "AddResult",
    "BotPerformanceHistoryDB",
]
