"""Database routing utilities for Menace."""

import sqlite3
import threading
import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

# Tables shared across Menace instances
SHARED_TABLES = {
    "enhancements",
    "bots",
    "errors",
    "code",
    "discrepancies",
    "workflow_summaries",
    # Additional shared tables can be added here
}

# Tables that are always local to a Menace instance
LOCAL_TABLES = {
    "models",
    "sandbox_metrics",
    "roi_logs",
    "menace_config",
    # Additional local tables can be added here
}


class DBRouter:
    """Route queries to the appropriate SQLite database."""

    def __init__(self, menace_id: str, local_db_path: str, shared_db_path: str) -> None:
        self.menace_id = menace_id
        self._local_conn = sqlite3.connect(local_db_path, check_same_thread=False)
        self._shared_conn = sqlite3.connect(shared_db_path, check_same_thread=False)
        self._local_lock = threading.Lock()
        self._shared_lock = threading.Lock()

    @contextmanager
    def get_connection(self, table_name: str) -> Iterator[sqlite3.Connection]:
        """Yield the appropriate connection for *table_name*.

        Tables listed in :data:`SHARED_TABLES` use the shared database while
        tables listed in :data:`LOCAL_TABLES` or unlisted tables default to the
        local database.
        """

        if table_name in SHARED_TABLES:
            logger.info("menace %s accessing shared table %s", self.menace_id, table_name)
            lock = self._shared_lock
            conn = self._shared_conn
        else:
            lock = self._local_lock
            conn = self._local_conn

        with lock:
            yield conn
