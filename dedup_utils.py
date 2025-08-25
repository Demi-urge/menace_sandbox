from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Sequence

__all__ = ["hash_fields", "insert_if_unique"]


def hash_fields(data: dict[str, Any], fields: Sequence[str]) -> str:
    """JSON-encode selected ``fields`` from ``data`` and return a SHA256 digest."""

    payload = {key: data[key] for key in fields}
    json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


# Alias to avoid shadowing by the ``hash_fields`` parameter in ``insert_if_unique``
_hash_fields = hash_fields


def insert_if_unique(
    conn: sqlite3.Connection,
    table: str,
    values: dict[str, Any],
    hash_fields: Sequence[str],
    menace_id: str,
    logger: Any,
) -> int | None:
    """Insert ``values`` into ``table`` unless a duplicate ``content_hash`` exists."""

    content_hash = _hash_fields(values, hash_fields)
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
        logger.warning("Duplicate insert ignored for %s (menace_id=%s)", table, menace_id)
        return None
