from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List
import json
import threading

from dynamic_path_router import resolve_path
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("sandbox_runner_input_history_db")


@dataclass
class InputRecord:
    data: dict[str, Any]
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class InputHistoryDB:
    """Simple SQLite-backed store for sandbox input stubs."""

    def __init__(self, path: Path | str = "input_history.db") -> None:
        if isinstance(path, Path):
            self.path = path
        else:
            try:
                self.path = Path(resolve_path(path))
            except FileNotFoundError:
                self.path = Path(resolve_path(".")) / path
        self._lock = threading.Lock()
        conn = router.get_connection("history")
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT,
            ts TEXT
        )
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts)"
        )
        conn.commit()

    def add(self, rec: InputRecord | dict[str, Any]) -> None:
        if isinstance(rec, dict):
            rec = InputRecord(rec)
        payload = json.dumps(rec.data)
        with self._lock:
            conn = router.get_connection("history")
            conn.execute(
                "INSERT INTO history(data, ts) VALUES(?, ?)",
                (payload, rec.ts),
            )
            conn.commit()

    def sample(self, limit: int = 10) -> List[dict[str, Any]]:
        with self._lock:
            conn = router.get_connection("history")
            rows = conn.execute(
                "SELECT data FROM history ORDER BY RANDOM() LIMIT ?",
                (int(limit),),
            ).fetchall()
        samples: List[dict[str, Any]] = []
        for r in rows:
            try:
                obj = json.loads(r[0])
                if isinstance(obj, dict):
                    samples.append(obj)
            except Exception:
                continue
        return samples

    def recent(self, limit: int = 10) -> List[dict[str, Any]]:
        """Return up to ``limit`` most recent records."""
        with self._lock:
            conn = router.get_connection("history")
            rows = conn.execute(
                "SELECT data FROM history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        records: List[dict[str, Any]] = []
        for r in rows:
            try:
                obj = json.loads(r[0])
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue
        return records


__all__ = ["InputHistoryDB", "InputRecord"]
