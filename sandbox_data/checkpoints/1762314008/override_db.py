"""SQLite storage for auto-override policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .db_router import GLOBAL_ROUTER, init_db_router


@dataclass
class OverrideRecord:
    signature: str
    require_human: bool
    ts: str


class OverrideDB:
    """Manage persisted override flags."""

    def __init__(self, path: Path | str = "overrides.db") -> None:
        self.path = Path(path)
        self.router = GLOBAL_ROUTER or init_db_router(
            "overrides", str(self.path), str(self.path)
        )
        self.conn = self.router.get_connection("overrides")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overrides(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature TEXT UNIQUE,
                require_human INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def get(self, signature: str) -> Optional[OverrideRecord]:
        cur = self.conn.execute(
            "SELECT signature, require_human, ts FROM overrides WHERE signature=?",
            (signature,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return OverrideRecord(row[0], bool(row[1]), row[2])

    def set(self, signature: str, require_human: bool) -> None:
        rec = self.get(signature)
        ts = datetime.utcnow().isoformat()
        if rec is None:
            self.conn.execute(
                "INSERT INTO overrides(signature, require_human, ts) VALUES (?,?,?)",
                (signature, int(require_human), ts),
            )
        else:
            self.conn.execute(
                "UPDATE overrides SET require_human=?, ts=? WHERE signature=?",
                (int(require_human), ts, signature),
            )
        self.conn.commit()

    def all(self) -> List[OverrideRecord]:
        cur = self.conn.execute(
            "SELECT signature, require_human, ts FROM overrides"
        )
        return [OverrideRecord(r[0], bool(r[1]), r[2]) for r in cur.fetchall()]

