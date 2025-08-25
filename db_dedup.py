from __future__ import annotations

"""Helpers for avoiding duplicate database rows.

The :func:`insert_if_unique` function computes a ``content_hash`` for selected
fields and attempts to insert a row into a table.  If a duplicate hash is
detected via a ``UNIQUE`` constraint the insert is skipped and the existing
row's ID is returned.
"""

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any

from db_router import SHARED_TABLES, DBRouter

try:  # pragma: no cover - optional dependency
    from sqlalchemy.exc import IntegrityError as SAIntegrityError
except Exception:  # pragma: no cover - optional dependency
    class SAIntegrityError(Exception):
        """Fallback when SQLAlchemy isn't installed."""

        pass

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine

__all__ = [
    "compute_content_hash",
    "hash_fields",
    "insert_if_unique",
    "ensure_content_hash_column",
]


def _sort_nested(value: Any) -> Any:
    """Recursively sort lists and tuples within ``value``.

    Dictionaries are left as-is since ``json.dumps`` with ``sort_keys=True``
    will handle key ordering. Lists and tuples are normalised by sorting their
    elements after recursively applying this function. Elements are sorted using
    their JSON representation to provide a deterministic order even for nested
    structures.
    """

    if isinstance(value, Mapping):
        return {k: _sort_nested(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return sorted(
            (_sort_nested(v) for v in value),
            key=lambda x: json.dumps(x, sort_keys=True),
        )
    return value


def compute_content_hash(data: Mapping[str, Any]) -> str:
    """Return a SHA256 hex digest for ``data``.

    The mapping is JSON encoded with keys sorted to ensure stable hashes for
    logically equivalent inputs. Lists and tuples contained within the mapping
    are sorted recursively prior to encoding so that their ordering does not
    affect the resulting hash.
    """

    normalized = _sort_nested(data)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True).encode("utf-8")
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
    queue_path: str | Path | None = None,
    router: DBRouter | None = None,
) -> Any | None:
    """Insert ``values`` into ``table`` if their hash is unique.

    ``hash_fields`` specifies which keys from ``values`` are hashed using
    :func:`compute_content_hash` to detect duplicates.  The resulting
    ``content_hash`` is added to the values prior to insertion.  If the hash
    already exists the insert is skipped and the existing row's ID is
    returned.

    Supply ``engine`` with a SQLAlchemy :class:`~sqlalchemy.Table` for SQL
    databases or ``conn`` with a table name for SQLite connections.  When
    ``queue_path`` is provided or the ``USE_DB_QUEUE`` environment variable is
    set, writes are queued via :func:`db_write_queue.queue_insert` instead of
    executing immediately.
    """

    table_name = table.name if hasattr(table, "name") else str(table)

    if router is not None and table_name in SHARED_TABLES:
        router.queue_write(table_name, values, hash_fields)
        return None

    use_queue = queue_path is not None or bool(os.getenv("USE_DB_QUEUE"))
    if use_queue:
        from db_write_queue import queue_insert  # avoid circular import

        queue_insert(table_name, values, hash_fields, queue_path)
        return None

    missing = [key for key in hash_fields if key not in values]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing fields for hashing: {missing_list}")

    payload = {key: values[key] for key in hash_fields}
    content_hash = compute_content_hash(payload)
    values = dict(values)
    values["content_hash"] = content_hash

    if engine is not None:
        columns = ", ".join(values.keys())
        placeholders = ", ".join(f":{k}" for k in values)
        from sqlalchemy import select, text  # type: ignore

        with engine.begin() as eng_conn:
            if engine.dialect.name == "sqlite":
                version = sqlite3.sqlite_version_info
                pk_col = list(table.primary_key.columns)[0].name
                if version >= (3, 35, 0):
                    sql = (
                        f"INSERT INTO {table.name} ({columns}) VALUES ({placeholders}) "
                        f"ON CONFLICT(content_hash) DO NOTHING RETURNING {pk_col}"
                    )
                    row = eng_conn.execute(text(sql), values).fetchone()
                    if row:
                        return row[0]
                else:
                    sql = (
                        f"INSERT OR IGNORE INTO {table.name} ({columns}) VALUES ({placeholders})"
                    )
                    result = eng_conn.execute(text(sql), values)
                    if result.rowcount:
                        return result.lastrowid

                logger.warning(
                    "Duplicate insert ignored for %s (menace_id=%s)",
                    table.name,
                    menace_id,
                )
                row = eng_conn.execute(
                    text(
                        f"SELECT {pk_col} FROM {table.name} WHERE content_hash=:hash"
                    ),
                    {"hash": content_hash},
                ).fetchone()
                return row[0] if row else None

            pk_col = list(table.primary_key.columns)[0]
            stmt = table.insert().values(**values)
            if hasattr(stmt, "on_conflict_do_nothing"):
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["content_hash"]
                ).returning(pk_col)
                row = eng_conn.execute(stmt).fetchone()
                if row:
                    return row[0]
                logger.warning(
                    "Duplicate insert ignored for %s (menace_id=%s)",
                    table.name,
                    menace_id,
                )
                row = eng_conn.execute(
                    select(pk_col).where(table.c.content_hash == content_hash)
                ).fetchone()
                return row[0] if row else None
            raise TypeError(
                "Unsupported engine dialect without conflict handling"
            )

    if conn is not None:
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        version = sqlite3.sqlite_version_info
        if version >= (3, 35, 0):
            cur = conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
                "ON CONFLICT(content_hash) DO NOTHING RETURNING id",
                tuple(values.values()),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])
        else:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            if cur.rowcount:
                return int(cur.lastrowid)

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


def ensure_content_hash_column(
    table: str,
    *,
    engine: "Engine | None" = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Ensure ``table`` has a ``content_hash`` column and unique index.

    The helper works with either a SQLAlchemy ``engine`` or a raw SQLite
    ``conn``.  If the table does not exist the function is a no-op.  When the
    column is added a unique index named ``idx_<table>_content_hash`` is also
    created to enforce uniqueness consistently across the code base.
    """

    if engine is not None:
        with engine.begin() as eng_conn:
            exists = eng_conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                return
            cols = eng_conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            if "content_hash" not in [c[1] for c in cols]:
                eng_conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN content_hash TEXT NOT NULL"
                )
            eng_conn.exec_driver_sql(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_content_hash"
                f" ON {table}(content_hash)"
            )
        return

    if conn is not None:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            return
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if "content_hash" not in [c[1] for c in cols]:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN content_hash TEXT NOT NULL"
            )
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_content_hash"
            f" ON {table}(content_hash)"
        )
        conn.commit()
        return

    raise TypeError("Either 'engine' or 'conn' must be provided")
