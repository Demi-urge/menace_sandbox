from __future__ import annotations

"""SQLite backed ledger for Stripe billing events.

This module exposes :class:`StripeLedger` which provides a :meth:`log_event`
method for persisting billing actions.  A module level :data:`STRIPE_LEDGER`
instance is exported along with a convenience :func:`log_event` wrapper so other
modules can easily record Stripe events.
"""

from pathlib import Path
from typing import Optional

from db_router import DBRouter

try:  # resolve paths dynamically when available
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - optional dependency
    resolve_path = None  # type: ignore

__all__ = ["StripeLedger", "STRIPE_LEDGER", "log_event"]


# Determine the storage location for the ledger database
_LOG_DIR = resolve_path("finance_logs") if resolve_path else Path("finance_logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _LOG_DIR / "stripe_ledger.db"
_MAX_BYTES = 5 * 1024 * 1024  # 5MB threshold for compaction/rotation


class StripeLedger:
    """Lightweight helper around an SQLite ledger database."""

    def __init__(self, db_path: Path = _DB_PATH, max_bytes: int = _MAX_BYTES) -> None:
        self.db_path = Path(db_path)
        self.max_bytes = max_bytes
        self._prepare_storage()
        # Use ``DBRouter`` so connections are wrapped like the rest of the codebase
        self.router = DBRouter("stripe_ledger", str(self.db_path), str(self.db_path))
        self._init_table()

    # ------------------------------------------------------------------
    def _prepare_storage(self) -> None:
        """Ensure directories exist and rotate oversized logs."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists() and self.db_path.stat().st_size > self.max_bytes:
            backup = self.db_path.with_suffix(self.db_path.suffix + ".1")
            try:
                self.db_path.replace(backup)
            except OSError:
                pass  # best effort; failing to rotate isn't fatal

    # ------------------------------------------------------------------
    def _init_table(self) -> None:
        conn = self.router.get_connection("stripe_ledger", "write")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stripe_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                bot_id TEXT,
                amount REAL,
                currency TEXT,
                user_email TEXT,
                account_id TEXT,
                timestamp INTEGER
            )
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    def _maybe_compact(self) -> None:
        """Compact the database when exceeding ``max_bytes``."""

        if self.db_path.exists() and self.db_path.stat().st_size > self.max_bytes:
            conn = self.router.get_connection("stripe_ledger", "write")
            conn.execute("VACUUM")
            conn.commit()

    # ------------------------------------------------------------------
    def log_event(
        self,
        action: str,
        bot_id: str,
        amount: float,
        currency: str,
        email: Optional[str],
        account_id: str,
        ts: int,
    ) -> int:
        """Insert a billing event and return the new row id."""

        conn = self.router.get_connection("stripe_ledger", "write")
        cursor = conn.execute(
            """
            INSERT INTO stripe_ledger (
                action, bot_id, amount, currency, user_email, account_id, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (action, bot_id, amount, currency, email, account_id, ts),
        )
        conn.commit()
        self._maybe_compact()
        return int(cursor.lastrowid)


# Module level singleton for reuse
STRIPE_LEDGER = StripeLedger()


def log_event(
    action: str,
    bot_id: str,
    amount: float,
    currency: str,
    email: Optional[str],
    account_id: str,
    ts: int,
) -> int:
    """Convenience wrapper around :class:`StripeLedger.log_event`."""

    return STRIPE_LEDGER.log_event(action, bot_id, amount, currency, email, account_id, ts)
