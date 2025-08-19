"""Simple on-disk cache for CLI retrieval results.

Stores (query, dbs) -> results mappings with associated database
modification times. Cached entries are invalidated automatically when any
underlying database file changes.
"""

from __future__ import annotations

import json
import os
import shelve
from pathlib import Path
from typing import Iterable, Any, Dict

__all__ = ["RetrievalCache", "get_db_mtimes"]


class RetrievalCache:
    """Tiny shelve-backed cache for retrieval results."""

    def __init__(self, path: str | Path = ".retrieval_cache.shelve", max_entries: int = 128) -> None:
        self.path = str(path)
        self.max_entries = max_entries

    def _key(self, query: str, dbs: Iterable[str]) -> str:
        return json.dumps({"q": query, "dbs": sorted(dbs)}, sort_keys=True)

    def get(self, query: str, dbs: Iterable[str], mtimes: Dict[str, float]) -> list[Any] | None:
        """Return cached results if available and valid."""
        key = self._key(query, dbs)
        with shelve.open(self.path) as sh:
            entry = sh.get(key)
            if not entry:
                return None
            cached_mtimes = entry.get("mtimes", {})
            for name, mtime in mtimes.items():
                if cached_mtimes.get(name) != mtime:
                    return None
            return entry.get("results")

    def set(self, query: str, dbs: Iterable[str], results: list[Any], mtimes: Dict[str, float]) -> None:
        """Store results along with database modification times."""
        key = self._key(query, dbs)
        with shelve.open(self.path, writeback=True) as sh:
            sh[key] = {"results": results, "mtimes": dict(mtimes), "ts": os.path.getmtime(self.path) if os.path.exists(self.path) else 0}
            # enforce max size by removing oldest entries
            if self.max_entries and len(sh) > self.max_entries:
                oldest_key = min(sh, key=lambda k: sh[k].get("ts", 0))
                del sh[oldest_key]


def get_db_mtimes(dbs: Iterable[str] | None) -> Dict[str, float]:
    """Return modification times for the given database names.

    The function attempts to locate SQLite files named ``"<db>.db"`` and also
    honours ``<DB>_DB_PATH`` environment variables. Missing files simply return
    a timestamp of ``0`` so cache entries will be invalidated once the database
    is created.
    """
    mtimes: Dict[str, float] = {}
    for name in sorted(set(dbs or [])):
        env_var = f"{name.upper()}_DB_PATH"
        path = Path(os.getenv(env_var, f"{name}.db"))
        try:
            mtimes[name] = path.stat().st_mtime
        except OSError:
            mtimes[name] = 0.0
    return mtimes
