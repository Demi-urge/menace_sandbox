"""Database routing utilities for Menace.

This module exposes a :class:`DBRouter` that decides whether a table should
reside in the local or the shared SQLite database.  Shared tables are
available to every Menace instance while local tables are isolated per
``menace_id``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Set

__all__ = ["DBRouter", "SHARED_TABLES", "LOCAL_TABLES", "init_db_router", "GLOBAL_ROUTER"]


# Tables stored in the shared database.  These tables are visible to every
# Menace instance.
SHARED_TABLES: Set[str] = {
    "enhancements",
    "bots",
    "errors",
    "code",
    "discrepancies",
    "workflow_summaries",
}

# Tables stored in the local database.  These are private to a specific
# ``menace_id`` instance.
LOCAL_TABLES: Set[str] = {
    "models",
    "patch_history",
    "variants",
    "memory",
    "events",
}


# Global router instance used by modules that rely on a single router without
# passing it around explicitly.  ``init_db_router`` populates this value.
GLOBAL_ROUTER: "DBRouter" | None = None


class DBRouter:
    """Route table operations to local or shared SQLite databases."""

    def __init__(self, menace_id: str, local_db_path: str, shared_db_path: str) -> None:
        """Create a new :class:`DBRouter`.

        Parameters
        ----------
        menace_id:
            Identifier for the Menace instance.  When ``local_db_path`` points to
            a directory a database file named ``"<menace_id>.db"`` will be created
            inside that directory.
        local_db_path:
            Path to the SQLite database used for local tables or a directory in
            which a database for ``menace_id`` should be created.
        shared_db_path:
            Path to the SQLite database used for shared tables.
        """

        self.menace_id = menace_id
        # When ``local_db_path`` is a directory, create a database file for this
        # menace instance inside it.
        local_path = (
            os.path.join(local_db_path, f"{menace_id}.db")
            if os.path.isdir(local_db_path)
            else local_db_path
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.local_conn = sqlite3.connect(local_path, check_same_thread=False)

        os.makedirs(os.path.dirname(shared_db_path), exist_ok=True)
        self.shared_conn = sqlite3.connect(shared_db_path, check_same_thread=False)

        # ``threading.Lock`` protects against concurrent access when deciding
        # which connection to return.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def get_connection(self, table_name: str) -> sqlite3.Connection:
        """Return the appropriate connection for ``table_name``.

        A :class:`ValueError` is raised for unknown tables.  Every request for a
        shared table is logged for observability.
        """

        if not table_name:
            raise ValueError("table_name must be a non-empty string")

        with self._lock:
            if table_name in SHARED_TABLES:
                logging.info("Routing table '%s' to shared database", table_name)
                return self.shared_conn
            if table_name in LOCAL_TABLES:
                return self.local_conn

        raise ValueError(f"Unknown table: {table_name}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close both the local and shared database connections."""

        self.local_conn.close()
        self.shared_conn.close()


def init_db_router(
    menace_id: str,
    local_db_path: str | None = None,
    shared_db_path: str | None = None,
) -> DBRouter:
    """Initialise a global :class:`DBRouter` instance.

    ``local_db_path`` defaults to ``./menace_<id>_local.db`` and
    ``shared_db_path`` defaults to ``./shared/global.db`` when not provided.
    The created router is stored in :data:`GLOBAL_ROUTER` and returned.
    """

    global GLOBAL_ROUTER

    local_path = local_db_path or f"./menace_{menace_id}_local.db"
    shared_path = shared_db_path or "./shared/global.db"

    GLOBAL_ROUTER = DBRouter(menace_id, local_path, shared_path)
    return GLOBAL_ROUTER


