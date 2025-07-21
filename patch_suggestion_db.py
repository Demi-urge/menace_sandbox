from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Optional, List
import sqlite3
import threading
from filelock import FileLock


@dataclass
class SuggestionRecord:
    module: str
    description: str
    ts: str = datetime.utcnow().isoformat()


class PatchSuggestionDB:
    """Store successful patch descriptions per module."""

    def __init__(self, path: Path | str = "suggestions.db") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._file_lock = FileLock(str(self.path) + ".lock")
        with self._file_lock:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
            CREATE TABLE IF NOT EXISTS suggestions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT,
                description TEXT,
                ts TEXT
            )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_suggestions_module ON suggestions(module)"
            )
            conn.commit()

    def add(self, rec: SuggestionRecord) -> None:
        with self._file_lock:
            with self._lock:
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "INSERT INTO suggestions(module, description, ts) VALUES(?,?,?)",
                        (rec.module, rec.description, rec.ts),
                    )
                    conn.commit()

    def history(self, module: str, limit: int = 10) -> List[str]:
        with self._lock:
            with sqlite3.connect(self.path) as conn:
                rows = conn.execute(
                    "SELECT description FROM suggestions WHERE module=? ORDER BY id DESC LIMIT ?",
                    (module, limit),
                ).fetchall()
        return [r[0] for r in rows]

    def best_match(self, module: str) -> Optional[str]:
        past = self.history(module, limit=20)
        if not past:
            return None
        desc, _ = Counter(past).most_common(1)[0]
        return desc


__all__ = ["PatchSuggestionDB", "SuggestionRecord"]
