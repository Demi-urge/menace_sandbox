from __future__ import annotations

"""Log Stripe billing events to a database with file fallback."""

import json
import logging
from pathlib import Path
from typing import Any

from db_router import GLOBAL_ROUTER, init_db_router
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)

try:
    _log_dir = resolve_path("finance_logs")
except FileNotFoundError:
    _log_dir = Path("finance_logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_LEDGER_FILE = _log_dir / "stripe_ledger.jsonl"

_COLUMNS = [
    "id",
    "action_type",
    "amount",
    "currency",
    "timestamp_ms",
    "user_email",
    "bot_id",
    "destination_account",
    "charge_id",
    "raw_event_json",
    "error",
]


def _get_connection():
    """Return connection to the ``stripe_ledger`` table or ``None``."""

    router = GLOBAL_ROUTER or init_db_router("default")
    try:
        conn = router.get_connection("stripe_ledger")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stripe_ledger (
                id TEXT PRIMARY KEY,
                action_type TEXT,
                amount REAL,
                currency TEXT,
                timestamp_ms INTEGER,
                user_email TEXT,
                bot_id TEXT,
                destination_account TEXT,
                charge_id TEXT,
                raw_event_json TEXT,
                error INTEGER
            )
            """
        )
        conn.commit()
        # Migration for existing tables without charge_id
        cur = conn.execute("PRAGMA table_info(stripe_ledger)")
        cols = {row[1] for row in cur.fetchall()}
        if "charge_id" not in cols:
            conn.execute("ALTER TABLE stripe_ledger ADD COLUMN charge_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stripe_ledger_charge_id ON stripe_ledger(charge_id)"
        )
        conn.commit()
        return conn
    except Exception as exc:  # pragma: no cover - DB failures
        logger.exception("stripe_ledger connection failed: %s", exc)
        return None


def _append_to_file(record: dict[str, Any]) -> None:
    """Append *record* to the JSONL ledger file."""

    try:
        with _LEDGER_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # pragma: no cover - filesystem issues
        logger.exception("failed writing stripe ledger file: %s", exc)


def log_event(**kwargs: Any) -> None:
    """Log a Stripe billing event.

    Parameters are accepted via ``kwargs`` and mapped to the predefined schema
    columns. When the database is unavailable the event is appended to
    ``stripe_ledger.jsonl``.
    """

    record = {col: kwargs.get(col) for col in _COLUMNS}
    conn = _get_connection()
    if conn is not None:
        try:
            placeholders = ", ".join(["?"] * len(_COLUMNS))
            conn.execute(
                f"INSERT INTO stripe_ledger ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                tuple(record[col] for col in _COLUMNS),
            )
            conn.commit()
            return
        except Exception as exc:  # pragma: no cover - DB write issues
            logger.exception("stripe_ledger insert failed: %s", exc)
    _append_to_file(record)


__all__ = ["log_event"]
