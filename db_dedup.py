from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterable
from typing import Any

from db_router import DBRouter

__all__ = ["insert_if_unique"]


def insert_if_unique(
    table: str,
    values: dict[str, Any],
    hash_fields: Iterable[str],
    menace_id: str,
    router: DBRouter,
) -> bool:
    """Insert ``values`` into ``table`` if the content is unique.

    Parameters
    ----------
    table:
        Table name to insert into.
    values:
        Mapping of column names to values.  The mapping is mutated to include a
        ``content_hash`` entry.
    hash_fields:
        Iterable of keys from ``values`` whose contents should be hashed to
        detect duplicates.
    menace_id:
        Identifier of the menace instance performing the insert.  Used for log
        messages.
    router:
        :class:`DBRouter` used to obtain the database connection.

    Returns
    -------
    bool
        ``True`` if the row was inserted, ``False`` if a duplicate was
        detected.
    """

    hash_payload = {k: values[k] for k in hash_fields}
    content_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    values["content_hash"] = content_hash

    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)

    try:
        with router.get_connection(table, "write") as conn:
            conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
        return True
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc).upper():
            logging.warning(
                "Duplicate insert ignored for %s (menace_id=%s)", table, menace_id
            )
            return False
        raise
