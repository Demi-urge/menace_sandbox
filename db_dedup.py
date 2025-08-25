from __future__ import annotations

"""Helpers for avoiding duplicate database rows.

The :func:`insert_if_unique` function computes a ``content_hash`` for selected
fields and attempts to insert a row into a table.  If a duplicate hash is
detected via a ``UNIQUE`` constraint the insert is skipped.
"""

from collections.abc import Iterable, Mapping
import hashlib
import json
import sqlite3
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from sqlalchemy.exc import IntegrityError as SAIntegrityError
except Exception:  # pragma: no cover - optional dependency
    class SAIntegrityError(Exception):
        """Fallback when SQLAlchemy isn't installed."""

        pass

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine

__all__ = ["compute_content_hash", "hash_fields", "insert_if_unique"]


def compute_content_hash(data: Mapping[str, Any]) -> str:
    """Return a SHA256 hex digest for ``data``.

    The mapping is JSON encoded with keys sorted to ensure stable hashes for
    logically equivalent inputs.
    """

    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode("utf-8")
    ).hexdigest()


def hash_fields(data: Mapping[str, Any], fields: Iterable[str]) -> str:
    """JSON-encode selected ``fields`` from ``data`` and return a hash.

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
    table: "Table | str",
    values: Mapping[str, Any],
    hash_fields: Iterable[str],
    menace_id: str,
    *,
    logger: Any,
    engine: "Engine | None" = None,
    conn: sqlite3.Connection | None = None,
) -> Any | None:
    """Insert ``values`` into ``table`` if their hash is unique.

    ``hash_fields`` specifies which keys from ``values`` are hashed using
    :func:`compute_content_hash` to detect duplicates.  The resulting
    ``content_hash`` is added to the values prior to insertion.  If the hash
    already exists the insert is skipped and the existing row's ID is
    returned.

    Supply ``engine`` with a SQLAlchemy :class:`~sqlalchemy.Table` for SQL
    databases or ``conn`` with a table name for SQLite connections.
    """

    missing = [key for key in hash_fields if key not in values]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing fields for hashing: {missing_list}")

    payload = {key: values[key] for key in hash_fields}
    content_hash = compute_content_hash(payload)
    values = dict(values)
    values["content_hash"] = content_hash

    if engine is not None:
        try:
            with engine.begin() as eng_conn:
                result = eng_conn.execute(table.insert().values(**values))  # type: ignore[arg-type]
            return result.inserted_primary_key[0]
        except SAIntegrityError as exc:
            msg = str(getattr(exc, "orig", exc))
            if "content_hash" not in msg:
                raise
            logger.warning(
                "Duplicate insert ignored for %s (menace_id=%s)", table.name, menace_id
            )
            from sqlalchemy import select  # type: ignore

            with engine.begin() as eng_conn:
                pk_col = list(table.primary_key.columns)[0]
                row = eng_conn.execute(
                    select(pk_col).where(table.c.content_hash == content_hash)
                ).fetchone()
            return row[0] if row else None

    if conn is not None:
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

    raise TypeError("Either 'engine' or 'conn' must be provided")
