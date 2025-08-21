from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Optional, List
import sqlite3
import threading
from filelock import FileLock

from analysis.semantic_diff_filter import find_semantic_risks
from patch_safety import PatchSafety


@dataclass
class SuggestionRecord:
    module: str
    description: str
    ts: str = datetime.utcnow().isoformat()


class PatchSuggestionDB:
    """Store successful patch descriptions per module."""

    def __init__(
        self,
        path: Path | str = "suggestions.db",
        *,
        semantic_threshold: float = 0.5,
        safety: PatchSafety | None = None,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._file_lock = FileLock(str(self.path) + ".lock")
        self._semantic_threshold = semantic_threshold
        self._safety = safety
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
                conn.execute(
                    """
            CREATE TABLE IF NOT EXISTS decisions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT,
                description TEXT,
                accepted INTEGER,
                reason TEXT,
                ts TEXT
            )
            """
                )
                conn.commit()

    def add(self, rec: SuggestionRecord) -> None:
        risks = find_semantic_risks(
            rec.description.splitlines(), threshold=self._semantic_threshold
        )
        if risks:
            self.log_decision(
                rec.module, rec.description, False, risks[0][1], rec.ts
            )
            raise ValueError(f"unsafe suggestion: {risks[0][1]}")
        if self._safety and self._safety.is_risky(
            {"category": rec.description, "module": rec.module}
        ):
            self.log_decision(
                rec.module, rec.description, False, "failure similarity", rec.ts
            )
            raise ValueError("unsafe suggestion: failure similarity")
        with self._file_lock:
            with self._lock:
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "INSERT INTO suggestions(module, description, ts) VALUES(?,?,?)",
                        (rec.module, rec.description, rec.ts),
                    )
                    conn.commit()
        self.log_decision(rec.module, rec.description, True, "", rec.ts)

    def log_decision(
        self,
        module: str,
        description: str,
        accepted: bool,
        reason: str,
        ts: str | None = None,
    ) -> None:
        ts = ts or datetime.utcnow().isoformat()
        with self._file_lock:
            with self._lock:
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "INSERT INTO decisions(module, description, accepted, reason, ts) VALUES(?,?,?,?,?)",
                        (module, description, int(accepted), reason, ts),
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
