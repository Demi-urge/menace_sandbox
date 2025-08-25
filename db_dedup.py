from __future__ import annotations

"""Helpers for avoiding duplicate database rows.

The :func:`insert_if_unique` function computes a ``content_hash`` for selected
fields and attempts to insert a row into a table.  If a duplicate hash is
detected via a ``UNIQUE`` constraint the insert is skipped.
"""

from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from sqlalchemy.exc import IntegrityError
except Exception:  # pragma: no cover - optional dependency
    class IntegrityError(Exception):
        """Fallback when SQLAlchemy isn't installed."""

        pass

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine

__all__ = ["compute_content_hash", "insert_if_unique"]


def compute_content_hash(data: Mapping[str, Any]) -> str:
    """Return a SHA256 hex digest for ``data``.

    The mapping is JSON encoded with keys sorted to ensure stable hashes for
    logically equivalent inputs.
    """

    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode("utf-8")
    ).hexdigest()


def insert_if_unique(
    table: Table,
    values: Mapping[str, Any],
    hash_fields: Iterable[str],
    menace_id: str,
    *,
    engine: Engine,
    logger: Any,
) -> Any | None:
    """Insert ``values`` into ``table`` if their hash is unique.

    ``hash_fields`` specifies which keys from ``values`` are hashed using
    :func:`compute_content_hash` to detect duplicates.  The resulting
    ``content_hash`` is added to the values prior to insertion.  If the hash
    already exists the insert is skipped and ``None`` is returned.
    """

    import sqlalchemy as sa  # Imported lazily to avoid mandatory dependency

    payload = {key: values[key] for key in hash_fields}
    content_hash = compute_content_hash(payload)
    values = dict(values)
    values["content_hash"] = content_hash

    try:
        with engine.begin() as conn:
            result = conn.execute(table.insert().values(**values))
        return result.inserted_primary_key[0]
    except IntegrityError as exc:
        # Only treat as a duplicate if the ``content_hash`` constraint failed.
        msg = str(getattr(exc, "orig", exc))
        if "content_hash" not in msg:
            raise
        logger.warning(
            "Duplicate insert ignored for %s (menace_id=%s)", table.name, menace_id
        )
        return None
