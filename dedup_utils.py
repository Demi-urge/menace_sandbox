from __future__ import annotations

"""Utilities for deduplicating SQLite inserts."""

from collections.abc import Iterable, Mapping
import sqlite3
from typing import Any

from db_dedup import compute_content_hash

__all__ = ["hash_fields", "insert_if_unique"]


def hash_fields(data: Mapping[str, Any], fields: Iterable[str]) -> str:
    """Return a hash of selected ``fields`` from ``data``.

    Raises
    ------
    KeyError
        If any of ``fields`` are missing from ``data``.
    """

    missing = [key for key in fields if key not in data]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing fields for hashing: {missing_list}")

    payload = {key: data[key] for key in fields}
    return compute_content_hash(payload)


def insert_if_unique(
    table: str,
    values: Mapping[str, Any],
    hash_fields: Iterable[str],
    menace_id: str,
    *,
    logger: Any,
    conn: sqlite3.Connection,
) -> int | None:
    """Insert ``values`` into ``table`` if their hash is unique.

    ``hash_fields`` specifies which keys from ``values`` are hashed using
    :func:`compute_content_hash` to detect duplicates.  The resulting
    ``content_hash`` is added to the values prior to insertion.  If the hash
    already exists the insert is skipped and the existing row's ID is
    returned.
    """

    missing = [key for key in hash_fields if key not in values]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing fields for hashing: {missing_list}")

    payload = {key: values[key] for key in hash_fields}
    content_hash = compute_content_hash(payload)
    values = dict(values)
    values["content_hash"] = content_hash

    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)
    try:
        cur = conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        logger.warning(
            "Duplicate insert ignored for %s (menace_id=%s)", table, menace_id
        )
        cur = conn.execute(
            f"SELECT id FROM {table} WHERE content_hash=?",
            (content_hash,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
