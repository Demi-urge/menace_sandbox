"""Convenience helpers for interacting with the Menace database."""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from .databases import MenaceDB

DEFAULT_DB_URL = "sqlite:///menace.db"


def connect(url: Optional[str] = None, queue_path: str | Path | None = None) -> MenaceDB:
    """Return a :class:`MenaceDB` instance using the provided or default URL."""

    return MenaceDB(url or DEFAULT_DB_URL, queue_path=queue_path)


def initialize(url: Optional[str] = None, queue_path: str | Path | None = None) -> MenaceDB:
    """Create tables and return a connected database."""

    db = connect(url, queue_path)
    if not db.use_queue:
        db.meta.create_all(db.engine)
    return db


__all__ = ["MenaceDB", "connect", "initialize"]
