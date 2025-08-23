"""SQLite database router for Menace.

This module provides a :class:`DBRouter` that routes table operations to either
a local database specific to a Menace instance or a shared global database.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Optional, Set

__all__ = ["DBRouter", "SHARED_TABLES", "LOCAL_TABLES"]


# Tables that should always be stored in the shared database.
SHARED_TABLES: Set[str] = {
    "enhancements",
    "bots",
    "errors",
    "code",
    "discrepancies",
    "workflow_summaries",
}

# Tables that are explicitly stored in the local database.
LOCAL_TABLES: Set[str] = {
    "models",
    "patch_history",
    "variants",
    "memory",
    "events",
}


class DBRouter:
    """Route table operations to local or shared SQLite databases."""

    def __init__(
        self,
        menace_id: str,
        local_db_path: str,
        shared_db_path: str,
        logger: Optional[logging.Logger] = None,
        log_level: int | None = logging.INFO,
    ) -> None:
        """Create a new :class:`DBRouter`.

        Parameters
        ----------
        menace_id:
            Identifier for the Menace instance.
        local_db_path:
            Path to the SQLite database used for local tables.
        shared_db_path:
            Path to the SQLite database used for shared tables.
        """

        self.menace_id = menace_id
        local_path = (
            os.path.join(local_db_path, f"{menace_id}.db")
            if os.path.isdir(local_db_path)
            else local_db_path
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.local_conn = sqlite3.connect(local_path, check_same_thread=False)
        os.makedirs(os.path.dirname(shared_db_path), exist_ok=True)
        self.shared_conn = sqlite3.connect(shared_db_path, check_same_thread=False)
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
        self._lock = threading.Lock()

    def get_connection(self, table_name: str) -> sqlite3.Connection:
        """Return a connection for ``table_name``.

        The table must exist in :data:`SHARED_TABLES` or :data:`LOCAL_TABLES`.
        A :class:`ValueError` is raised for unknown tables.
        """

        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty string")

        with self._lock:
            if table_name in SHARED_TABLES:
                conn = self.shared_conn
                source = "shared"
            elif table_name in LOCAL_TABLES:
                conn = self.local_conn
                source = "local"
            else:
                raise ValueError(f"Unknown table: {table_name}")

        if self.log_level is not None and self.logger:
            self.logger.log(
                self.log_level,
                "get_connection called for table '%s'; returning %s connection",
                table_name,
                source,
            )

        return conn

    def close(self) -> None:
        """Close both the local and shared database connections."""

        self.local_conn.close()
        self.shared_conn.close()

