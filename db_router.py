"""SQLite database router for Menace.

This module provides a :class:`DBRouter` that routes table operations to either
a local database specific to a Menace instance or a shared global database.
"""

from __future__ import annotations

import sqlite3
from typing import Set

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

    def __init__(self, menace_id: str, local_db_path: str, shared_db_path: str) -> None:
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
        self.local_conn = sqlite3.connect(local_db_path, check_same_thread=False)
        self.shared_conn = sqlite3.connect(shared_db_path, check_same_thread=False)

    def get_connection(self, table_name: str) -> sqlite3.Connection:
        """Return a connection for ``table_name``.

        The table must exist in :data:`SHARED_TABLES` or :data:`LOCAL_TABLES`.
        A :class:`ValueError` is raised for unknown tables.
        """

        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty string")

        if table_name in SHARED_TABLES:
            return self.shared_conn

        if table_name in LOCAL_TABLES:
            return self.local_conn

        raise ValueError(f"Unknown table: {table_name}")

    def close(self) -> None:
        """Close both the local and shared database connections."""

        self.local_conn.close()
        self.shared_conn.close()

