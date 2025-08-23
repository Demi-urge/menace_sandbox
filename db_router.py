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

__all__ = ["DBRouter", "SHARED_TABLES"]


# Tables that should always be stored in the shared database.
SHARED_TABLES = {
    "enhancements",
    "bots",
    "errors",
    "code",
    "discrepancies",
    "workflow_summaries",
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

    @contextmanager
    def get_connection(self, table_name: str) -> Iterator[sqlite3.Connection]:
        """Return a connection for *table_name*.

        The router validates ``table_name`` and routes the request to the shared
        database if the name appears in :data:`SHARED_TABLES`; otherwise the
        request is routed to the local database. Routing decisions are logged and
        operations on the connection are protected by :pyattr:`lock`.
        """

        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")

        if table_name in SHARED_TABLES:
            logging.info("menace %s routing %s to shared database", self.menace_id, table_name)
            conn = self.shared_conn
        else:
            logging.info("menace %s routing %s to local database", self.menace_id, table_name)
            conn = self.local_conn

        with self.lock:
            yield conn

    def close(self) -> None:
        """Close both the local and shared database connections."""

        self.local_conn.close()
        self.shared_conn.close()

