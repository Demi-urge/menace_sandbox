from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any, Tuple

__all__ = ["compute_content_hash", "insert_if_unique"]


def compute_content_hash(data: Mapping[str, Any]) -> str:
    """Return a SHA256 hash of ``data`` encoded as sorted JSON.

    Parameters
    ----------
    data:
        Mapping of field names to values that should be hashed.

    Returns
    -------
    str
        Hex digest of the SHA256 hash of the JSON representation of ``data``.
    """

    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode("utf-8")
    ).hexdigest()


def insert_if_unique(
    conn: sqlite3.Connection,
    table: str,
    values: Mapping[str, Any],
    hash_fields: Iterable[str],
    menace_id: str,
) -> Tuple[int, bool]:
    """Insert ``values`` into ``table`` if the content is unique.

    The hash of ``hash_fields`` is computed using :func:`compute_content_hash` and
    stored in the ``content_hash`` column.  If an existing row with the same
    ``content_hash`` is found, the insert is skipped and the existing row's ``id``
    is returned.

    Parameters
    ----------
    conn:
        SQLite connection used to execute the insert.
    table:
        Table name to insert into.
    values:
        Mapping of column names to values.  ``content_hash`` is added to this
        mapping before the insert.
    hash_fields:
        Iterable of keys from ``values`` whose contents should be hashed to
        detect duplicates.
    menace_id:
        Identifier of the menace instance performing the insert.  Used for log
        messages.

    Returns
    -------
    tuple[int, bool]
        A pair of ``(row_id, inserted)`` where ``row_id`` is the primary key of
        the existing or newly inserted row and ``inserted`` is ``True`` if a new
        row was created.
    """

    hash_payload = {k: values[k] for k in hash_fields}
    content_hash = compute_content_hash(hash_payload)
    values_with_hash = dict(values)
    values_with_hash["content_hash"] = content_hash

    columns = ", ".join(values_with_hash.keys())
    placeholders = ", ".join("?" for _ in values_with_hash)

    try:
        cur = conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values_with_hash.values()),
        )
        return int(cur.lastrowid), True
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc).upper():
            logging.warning(
                "Duplicate insert ignored for %s (menace_id=%s)", table, menace_id
            )
            cur = conn.execute(
                f"SELECT id FROM {table} WHERE content_hash=?",
                (content_hash,),
            )
            row = cur.fetchone()
            if row is None:
                raise
            return int(row[0]), False
        raise
