from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Mapping, Sequence

__all__ = ["hash_fields", "insert_if_unique"]


def hash_fields(data: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Return SHA256 hex digest of selected fields from ``data``.

    Parameters
    ----------
    data:
        Mapping containing the data to hash.
    fields:
        Sequence of keys from ``data`` whose values should be included in the
        hash calculation.
    """

    payload = {key: data[key] for key in fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


# Alias to avoid shadowing by the ``hash_fields`` parameter in insert_if_unique
_hash_fields = hash_fields


def insert_if_unique(
    conn: sqlite3.Connection,
    table: str,
    values: Mapping[str, Any],
    hash_fields: Sequence[str],
    menace_id: str,
    logger: Any,
) -> int | None:
    """Insert ``values`` into ``table`` if the hashed content is unique.

    ``hash_fields`` specifies which keys from ``values`` are hashed using
    :func:`hash_fields` to detect duplicates.  The resulting ``content_hash`` is
    inserted into the ``content_hash`` column.

    Parameters
    ----------
    conn:
        SQLite connection used for executing the insert.
    table:
        Table name to insert into.
    values:
        Mapping of column names to values.  ``content_hash`` is added to this
        mapping prior to insertion.
    hash_fields:
        Sequence of keys from ``values`` to hash for deduplication.
    menace_id:
        Identifier of the menace instance performing the insert, included in
        log messages.
    logger:
        Logger used for warning messages when duplicates are detected.

    Returns
    -------
    int | None
        ``lastrowid`` of the inserted row or ``None`` if a duplicate was
        detected.
    """

    content_hash = _hash_fields(values, hash_fields)
    values_with_hash = dict(values)
    values_with_hash["content_hash"] = content_hash

    columns = ", ".join(values_with_hash.keys())
    placeholders = ", ".join("?" for _ in values_with_hash)

    try:
        cur = conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values_with_hash.values()),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        logger.warning("Duplicate insert ignored for %s (menace_id=%s)", table, menace_id)
        return None
