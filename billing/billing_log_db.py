from __future__ import annotations

"""SQLite backed billing log storage."""

from dataclasses import dataclass
import sqlite3
from pathlib import Path

try:  # pragma: no cover - optional dependency during bootstrap
    import stripe_billing_router  # noqa: F401
except Exception:  # pragma: no cover - best effort import
    stripe_billing_router = None  # type: ignore

try:  # pragma: no cover - resolve_path is optional
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - if dynamic_path_router missing
    resolve_path = None  # type: ignore

if resolve_path is not None:
    try:
        _DB_PATH = resolve_path("menace.db")
    except FileNotFoundError:  # pragma: no cover - DB may not exist yet
        _DB_PATH = Path("menace.db")
else:  # pragma: no cover - fallback path
    _DB_PATH = Path("menace.db")


@dataclass
class BillingEvent:
    """Record describing a billing action."""

    action: str
    bot_id: str | None = None
    amount: float | None = None
    currency: str | None = None
    user_email: str | None = None
    destination_account: str | None = None
    key_hash: str | None = None
    stripe_id: str | None = None
    ts: str | None = None  # ISO timestamp; defaults to CURRENT_TIMESTAMP


class BillingLogDB:
    """Lightweight helper for persisting billing events."""

    def __init__(self, path: str | Path = _DB_PATH) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS billing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                bot_id TEXT,
                amount REAL,
                currency TEXT,
                user_email TEXT,
                destination_account TEXT,
                key_hash TEXT,
                stripe_id TEXT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        self.conn.commit()
        # Migration: rename legacy stripe_key column to key_hash
        cur.execute("PRAGMA table_info(billing_logs)")
        cols = {row[1] for row in cur.fetchall()}
        if "stripe_key" in cols and "key_hash" not in cols:
            cur.execute("ALTER TABLE billing_logs RENAME COLUMN stripe_key TO key_hash")
            self.conn.commit()
        # Migration: add stripe_id column if missing
        cur.execute("PRAGMA table_info(billing_logs)")
        cols = {row[1] for row in cur.fetchall()}
        if "stripe_id" not in cols:
            cur.execute("ALTER TABLE billing_logs ADD COLUMN stripe_id TEXT")
            self.conn.commit()

    # ------------------------------------------------------------------
    def log(self, event: BillingEvent) -> int:
        """Insert *event* into the database and return row id."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO billing_logs(
                action, bot_id, amount, currency, user_email,
                destination_account, key_hash, stripe_id, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            (
                event.action,
                event.bot_id,
                event.amount,
                event.currency,
                event.user_email,
                event.destination_account,
                event.key_hash,
                event.stripe_id,
                event.ts,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)


# Module level singleton
_DEFAULT_DB = BillingLogDB()


def log_billing_event(
    action: str,
    *,
    bot_id: str | None = None,
    amount: float | None = None,
    currency: str | None = None,
    user_email: str | None = None,
    destination_account: str | None = None,
    key_hash: str | None = None,
    stripe_id: str | None = None,
    ts: str | None = None,
) -> int:
    """Insert a billing event into the ``billing_logs`` table."""

    event = BillingEvent(
        action=action,
        bot_id=bot_id,
        amount=amount,
        currency=currency,
        user_email=user_email,
        destination_account=destination_account,
        key_hash=key_hash,
        stripe_id=stripe_id,
        ts=ts,
    )
    return _DEFAULT_DB.log(event)


__all__ = ["BillingLogDB", "BillingEvent", "log_billing_event"]
