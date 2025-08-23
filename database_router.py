"""Deprecated module stub for backwards compatibility.

This module previously exposed the :class:`DatabaseRouter` with extensive
cross-database query helpers.  It has been superseded by :mod:`db_router`'s
:class:`DBRouter` which focuses on routing table operations between local and
shared SQLite databases.  The shim below re-exports :class:`DBRouter` and
provides minimal placeholders so legacy imports continue to work while emitting
a deprecation warning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Dict, Any

from db_router import (
    DBRouter,
    SHARED_TABLES,
    LOCAL_TABLES,
    DENY_TABLES,
    init_db_router,
    GLOBAL_ROUTER,
)

warnings.warn(
    "database_router.py is deprecated; use db_router.DBRouter instead",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class DBResult:
    """Placeholder return type for legacy ``query_all`` calls."""

    code: List[Dict[str, Any]]
    bots: List[Dict[str, Any]]
    info: List[Any]
    memory: List[Any]
    menace: List[Dict[str, Any]]


class DatabaseRouter(DBRouter):
    """Compatibility shim mapping to :class:`db_router.DBRouter`.

    Only ``query_all`` is provided for legacy callers and simply returns an
    empty :class:`DBResult`.  The rich query aggregation of the original router
    has been removed.
    """

    def query_all(self, term: str, **options: Any) -> DBResult:  # type: ignore[override]
        return DBResult(code=[], bots=[], info=[], memory=[], menace=[])


__all__ = [
    "DatabaseRouter",
    "DBRouter",
    "DBResult",
    "SHARED_TABLES",
    "LOCAL_TABLES",
    "DENY_TABLES",
    "init_db_router",
    "GLOBAL_ROUTER",
]
