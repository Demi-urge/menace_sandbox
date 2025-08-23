"""Simple SQLite database router for Menace.

This module provides a :class:`DBRouter` that directs table queries to either a
local database specific to a Menace instance or a shared global database. The
router maintains a single re-entrant lock to ensure thread safety when
connections are retrieved.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "DBRouter",
    "SHARED_TABLES",
    "LOCAL_TABLES",
    "GLOBAL_ROUTER",
    "init_db_router",
]


# Tables that should always be stored in the shared database.
SHARED_TABLES = {
    "enhancements",
    "bots",
    "errors",
    "code",
    "discrepancies",
    "workflow_summaries",
}

# Tables that are explicitly stored in the local database.
LOCAL_TABLES = {
    "models",
    "patch_history",
    "variants",
    "memory",
    "events",
}


class DBRouter:
    """Route table operations to local or shared SQLite databases.

    Parameters
    ----------
    menace_id:
        Identifier for the Menace instance.
    local_path:
        Directory where the local database should be created. The database file
        will be named ``menace_{menace_id}_local.db``. Defaults to the current
        directory.
    shared_path:
        Path to the shared SQLite database. Defaults to ``./shared/global.db``.
    """

    def __init__(
        self, menace_id: str, local_path: str = "./", shared_path: str = "./shared/global.db"
    ) -> None:
        self.menace_id = menace_id
        self.local_conn = sqlite3.connect(
            f"{local_path}/menace_{menace_id}_local.db", check_same_thread=False
        )
        self.shared_conn = sqlite3.connect(shared_path, check_same_thread=False)
        self.lock = threading.RLock()
        self.local_calls = 0
        self.shared_calls = 0

    @contextmanager
    def get_connection(self, table_name: str) -> Iterator[sqlite3.Connection]:
        """Return a connection for *table_name*.

        ``table_name`` must exist in :data:`SHARED_TABLES` or
        :data:`LOCAL_TABLES`; otherwise a :class:`ValueError` is raised. The
        router logs routing decisions and counts the number of calls to each
        database for basic auditing.
        """

        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")

        if table_name in SHARED_TABLES:
            self.shared_calls += 1
            logger.debug(
                "menace %s routing %s to shared database", self.menace_id, table_name
            )
            conn = self.shared_conn
        elif table_name in LOCAL_TABLES:
            self.local_calls += 1
            logger.debug(
                "menace %s routing %s to local database", self.menace_id, table_name
            )
            conn = self.local_conn
        else:
            logger.debug(
                "menace %s attempted access to unknown table %s", self.menace_id, table_name
            )
            raise ValueError(
                f"table_name '{table_name}' is not registered as shared or local"
            )

        with self.lock:
            yield conn

    def close(self) -> None:
        """Close both the local and shared database connections."""

        self.local_conn.close()
        self.shared_conn.close()


# Global router instance used throughout the application.
GLOBAL_ROUTER: DBRouter | None = None


def init_db_router(
    menace_id: str, local_path: str = "./", shared_path: str = "./shared/global.db"
) -> DBRouter:
    """Initialise :class:`DBRouter` and store it globally.

    Parameters
    ----------
    menace_id:
        Identifier for the Menace instance. A unique value should be supplied for
        each process to keep local databases isolated.
    local_path:
        Optional override for where the local database lives.
    shared_path:
        Optional override for the shared database location.
    """

    global GLOBAL_ROUTER
    GLOBAL_ROUTER = DBRouter(menace_id, local_path=local_path, shared_path=shared_path)
    return GLOBAL_ROUTER

