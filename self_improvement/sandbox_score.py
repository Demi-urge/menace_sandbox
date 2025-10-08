from __future__ import annotations

"""Helpers for retrieving sandbox performance scores."""

import logging
import sqlite3

try:  # pragma: no cover - optional dependency location
    from menace_sandbox.dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore

logger = logging.getLogger(__name__)


def get_latest_sandbox_score(path: str) -> float:
    """Return the most recent sandbox score stored in ``path``.

    The helper expects a SQLite database containing a ``scores`` table with a
    ``score`` column and an accompanying timestamp column.  When the database or
    table is missing a warning is logged and ``0.0`` is returned.
    """

    try:
        conn = sqlite3.connect(str(resolve_path(path)))
    except Exception:
        logger.warning("sandbox score database missing at %s", path)
        return 0.0
    try:
        cur = conn.execute(
            "SELECT score FROM scores ORDER BY timestamp DESC LIMIT 1"
        )
        row = cur.fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        logger.warning("scores table missing in sandbox score database %s", path)
        return 0.0
    finally:
        conn.close()


__all__ = ["get_latest_sandbox_score"]
