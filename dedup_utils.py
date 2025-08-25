from __future__ import annotations

"""Utilities for deduplicating SQLite inserts."""

from collections.abc import Iterable, Mapping
import hashlib
import json
import sqlite3
from typing import Any

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
