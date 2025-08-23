from __future__ import annotations

"""Session Vault storing browser identities in Redis with SQLite backup.

Entry points must ensure the global database router is initialised before
interacting with the vault.  When executed as a script this module initialises
the router with the identifier ``"session_vault"``.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("session_vault")

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore


@dataclass
class SessionData:
    """Information required to reuse a browser session."""

    cookies: Dict[str, Any]
    user_agent: str
    fingerprint: str
    last_seen: float
    success_count: int = 0
    fail_count: int = 0
    session_id: Optional[int] = None
    domain: str | None = None


class SessionVault:
    """Store and retrieve sessions using Redis for speed and SQLite for durability."""

    def __init__(self, path: str | None = None, redis_url: str = "redis://localhost:6379/0") -> None:
        self.conn = router.local_conn
        self._init_sqlite()
        self.redis = None
        if redis and self._connect_redis(redis_url):
            self.redis_url = redis_url

    def _connect_redis(self, url: str) -> bool:
        try:
            self.redis = redis.Redis.from_url(url, decode_responses=True)
            self.redis.ping()
            return True
        except Exception:
            self.redis = None
            return False

    # SQLite schema
    def _init_sqlite(self) -> None:
        conn = self.conn
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                cookies TEXT,
                user_agent TEXT,
                fingerprint TEXT,
                last_seen REAL,
                success_count INTEGER,
                fail_count INTEGER
            )
            """
        )
        conn.commit()

    def add(self, domain: str, data: SessionData) -> int:
        """Insert a new session."""
        data.last_seen = time.time()
        data.domain = domain
        conn = self.conn
        cur = conn.execute(
            """INSERT INTO sessions(domain,cookies,user_agent,fingerprint,last_seen,success_count,fail_count)
            VALUES(?,?,?,?,?,?,?)""",
            (
                domain,
                json.dumps(data.cookies),
                data.user_agent,
                data.fingerprint,
                data.last_seen,
                data.success_count,
                data.fail_count,
            ),
        )
        conn.commit()
        sid = int(cur.lastrowid)

        data.session_id = sid
        if self.redis:
            self.redis.hset(f"session:{sid}", mapping=asdict(data))
            self.redis.lpush(f"domain:{domain}", sid)
        return sid

    def count(self, domain: str | None = None) -> int:
        """Return number of stored sessions optionally filtered by *domain*."""
        conn = self.conn
        if domain is None:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE domain=?",
                (domain,),
            ).fetchone()
        return int(row[0]) if row else 0

    def _load_from_sqlite(self, sid: int) -> Optional[SessionData]:
        conn = self.conn
        row = conn.execute(
            "SELECT id,domain,cookies,user_agent,fingerprint,last_seen,success_count,fail_count FROM sessions WHERE id=?",
            (sid,),
        ).fetchone()
        if not row:
            return None
        return SessionData(
            cookies=json.loads(row[2]),
            user_agent=row[3],
            fingerprint=row[4],
            last_seen=row[5],
            success_count=row[6],
            fail_count=row[7],
            session_id=row[0],
            domain=row[1],
        )

    def get_least_recent(self, domain: str) -> Optional[SessionData]:
        """Return least recently used healthy session for a domain."""
        sid = None
        if self.redis:
            while True:
                sid_raw = self.redis.rpoplpush(f"domain:{domain}", f"domain:{domain}")
                if not sid_raw:
                    break
                data = self.redis.hgetall(f"session:{sid_raw}")
                if data:
                    sid = int(sid_raw)
                    break
        if sid is None:
            conn = self.conn
            row = conn.execute(
                "SELECT id FROM sessions WHERE domain=? ORDER BY last_seen ASC LIMIT 1",
                (domain,),
            ).fetchone()
            if row:
                sid = int(row[0])
        if sid is None:
            return None
        data = self._load_from_sqlite(sid)
        return data

    def report(self, sid: int, *, success: bool | None = None, captcha: bool = False, banned: bool = False) -> None:
        """Update metrics for a session."""
        data = self._load_from_sqlite(sid)
        if not data:
            return
        if success:
            data.success_count += 1
        if captcha or banned or success is False:
            data.fail_count += 1
        data.last_seen = time.time()
        conn = self.conn
        conn.execute(
            "UPDATE sessions SET last_seen=?, success_count=?, fail_count=? WHERE id=?",
            (data.last_seen, data.success_count, data.fail_count, sid),
        )
        conn.commit()
        if self.redis:
            self.redis.hset(f"session:{sid}", mapping=asdict(data))
            if not self.redis.lrem(f"domain:{data.domain}", 0, sid):
                self.redis.lpush(f"domain:{data.domain}", sid)

__all__ = ["SessionData", "SessionVault"]
