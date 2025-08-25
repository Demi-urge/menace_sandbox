from __future__ import annotations

"""Helpers for avoiding duplicate database rows.

This module exposes a small pair of helpers that make it easier to avoid
inserting duplicate rows into a database table.  A SHA256 hash of selected
values is stored in a ``content_hash`` column which should be declared as a
``UNIQUE`` field on the target table.
"""

from collections.abc import Iterable, Mapping
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from sqlalchemy.exc import IntegrityError
except Exception:  # pragma: no cover - optional dependency
    class IntegrityError(Exception):
        """Fallback when SQLAlchemy isn't installed."""

        pass

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from sqlalchemy import Table
    from sqlalchemy.engine import Connection

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
    conn: Connection,
    table: Table,
    values: Mapping[str, Any],
    hash_fields: Iterable[str],
    menace_id: str,
) -> Any | None:
    """Insert ``values`` into ``table`` if their hash is unique.

    ``hash_fields`` specifies which keys from ``values`` are hashed using
    :func:`compute_content_hash` to detect duplicates.  The resulting
    ``content_hash`` is added to the values prior to insertion.  If the hash
    already exists the insert is skipped and the existing primary key is
    returned.
    """

    logger = logging.getLogger(__name__)

    import sqlalchemy as sa  # Imported lazily to avoid mandatory dependency

    payload = {key: values[key] for key in hash_fields}
    content_hash = compute_content_hash(payload)
    values_with_hash = dict(values)
    values_with_hash["content_hash"] = content_hash

    try:
        result = conn.execute(table.insert().values(**values_with_hash))
        return result.inserted_primary_key[0]
    except IntegrityError as exc:
        # Only treat as a duplicate if the ``content_hash`` constraint failed.
        msg = str(getattr(exc, "orig", exc))
        if "content_hash" not in msg:
            raise
        logger.warning(
            "Duplicate insert ignored for %s (menace_id=%s)", table.name, menace_id
        )
        pk_col = list(table.primary_key.columns)[0]
        row = conn.execute(
            sa.select(pk_col).where(table.c.content_hash == content_hash)
        ).first()
        return row[0] if row else None
