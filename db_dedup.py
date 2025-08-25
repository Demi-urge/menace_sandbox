from __future__ import annotations

"""Database deduplication helpers.

This module provides a small utility for inserting rows only if their
content is unique.  Uniqueness is determined by hashing selected fields
using JSON encoding with sorted keys.  The resulting SHA256 hash is stored
in a ``content_hash`` column which should be declared as ``UNIQUE`` in the
target table.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

__all__ = ["compute_content_hash", "insert_if_unique"]


def compute_content_hash(values: Mapping[str, Any], fields: Iterable[str]) -> str:
    """Return SHA256 hex digest of ``fields`` from ``values``.

    Parameters
    ----------
    values:
        Mapping containing the data to hash.
    fields:
        Iterable of keys from ``values`` whose values should be included in the
        hash calculation.
    """

    payload = {key: values[key] for key in fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
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
    """Insert ``values`` into ``table`` if the hashed content is unique.

    The hash of ``hash_fields`` is computed using :func:`compute_content_hash`
    and stored in the ``content_hash`` column before attempting the insert.

    Parameters
    ----------
    table:
        SQLAlchemy ``Table`` object representing the target table.
    values:
        Mapping of column names to values. ``content_hash`` is added to a copy
        of this mapping prior to insertion.
    hash_fields:
        Iterable of keys from ``values`` to include in the hash calculation.
    menace_id:
        Identifier of the menace instance performing the insert, included in
        log messages.
    engine:
        SQLAlchemy ``Engine`` used to execute the insert.
    logger:
        Logger used for warning messages when duplicates are detected.

    Returns
    -------
    Any | None
        Primary key of the inserted row, or ``None`` if a duplicate was
        detected.
    """

    values_copy = dict(values)
    values_copy["content_hash"] = compute_content_hash(values_copy, hash_fields)

    try:
        with engine.begin() as conn:
            result = conn.execute(table.insert().values(**values_copy))
            return result.inserted_primary_key[0]
    except IntegrityError as exc:
        # Only treat the error as a duplicate if the unique ``content_hash``
        # constraint triggered it. Otherwise re-raise for visibility.
        msg = str(getattr(exc, "orig", exc))
        if "content_hash" not in msg:
            raise
        logger.warning(
            "Duplicate insert ignored for %s (menace_id=%s)", table.name, menace_id
        )
        return None
