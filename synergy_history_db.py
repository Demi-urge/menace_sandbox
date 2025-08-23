"""Utilities for persisting synergy history.

The module relies on the global database router.  When executed as a script it
initialises the router with the identifier ``"synergy_history_db"``.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from contextlib import contextmanager
import logging
from filelock import FileLock

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from . import RAISE_ERRORS

router = GLOBAL_ROUTER or init_db_router("synergy_history_db")

logger = logging.getLogger(__name__)


class HistoryParseError(RuntimeError):
    """Raised when synergy history parsing fails."""

CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS synergy_history ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " entry TEXT NOT NULL)"
)


def connect(path: str | Path | None = None, *, router: DBRouter | None = None) -> sqlite3.Connection:
    """Return a connection to the synergy history database.

    When ``path`` is provided a temporary router pointing at that file is
    created; otherwise the globally initialised router is used.
    """

    if router is None:
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            r = DBRouter("synergy_history_db", str(p), str(p))
        else:
            r = globals()["router"]
    else:
        r = router

    conn = r.get_connection("synergy_history")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_TABLE)
    conn.commit()
    return conn


@contextmanager
def connect_locked(path: str | Path | None = None, *, router: DBRouter | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection with an exclusive lock on the database file."""

    if path is not None:
        lock_path = Path(path).with_suffix(Path(path).suffix + ".lock")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path))
        with lock:
            conn = connect(path, router=router)
            yield conn
    else:
        conn = connect(router=router)
        yield conn


def load_history(path: str | Path | None = None, *, router: DBRouter | None = None) -> List[Dict[str, float]]:
    """Return synergy history entries from the SQLite database."""
    if path is not None and not Path(path).exists():
        return []
    try:
        conn = connect(path, router=router)
        rows = conn.execute(
            "SELECT entry FROM synergy_history ORDER BY id"
        ).fetchall()
        hist: List[Dict[str, float]] = []
        for (text,) in rows:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    hist.append({str(k): float(v) for k, v in data.items()})
            except Exception as exc:
                logger.exception("invalid history entry: %s", exc)
                if RAISE_ERRORS:
                    raise HistoryParseError(str(exc)) from exc
        return hist
    except Exception as exc:
        logger.exception("failed to load history: %s", exc)
        if RAISE_ERRORS:
            raise HistoryParseError(str(exc)) from exc
        return []


def insert_entry(conn: sqlite3.Connection, entry: Dict[str, float]) -> None:
    path = None
    try:
        path = conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:
        pass
    if path:
        lock = FileLock(str(path) + ".lock")
        with lock:
            conn.execute(
                "INSERT INTO synergy_history(entry) VALUES (?)", (json.dumps(entry),)
            )
            conn.commit()
    else:
        conn.execute(
            "INSERT INTO synergy_history(entry) VALUES (?)", (json.dumps(entry),)
        )
        conn.commit()


def fetch_all(conn: sqlite3.Connection) -> List[Dict[str, float]]:
    rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
    out: List[Dict[str, float]] = []
    for (text,) in rows:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                out.append({str(k): float(v) for k, v in data.items()})
        except Exception:
            continue
    return out


def fetch_after(conn: sqlite3.Connection, last_id: int) -> List[Tuple[int, Dict[str, float]]]:
    """Return entries with ``id`` greater than ``last_id``."""
    rows = conn.execute(
        "SELECT id, entry FROM synergy_history WHERE id > ? ORDER BY id",
        (int(last_id),),
    ).fetchall()
    out: List[Tuple[int, Dict[str, float]]] = []
    for row_id, text in rows:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                out.append(
                    (int(row_id), {str(k): float(v) for k, v in data.items()})
                )
        except Exception as exc:
            logger.exception("failed to parse history row %s: %s", row_id, exc)
            if RAISE_ERRORS:
                raise HistoryParseError(str(exc)) from exc
    return out


def fetch_latest(conn: sqlite3.Connection) -> Dict[str, float]:
    row = conn.execute(
        "SELECT entry FROM synergy_history ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    try:
        data = json.loads(row[0])
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception as exc:
        logger.exception("failed to parse latest history entry: %s", exc)
        if RAISE_ERRORS:
            raise HistoryParseError(str(exc)) from exc
    return {}


def migrate_json_to_db(json_path: str | Path, db_path: str | Path | None = None) -> None:
    jp = Path(json_path)
    if not jp.exists():
        return
    try:
        data = json.loads(jp.read_text())
        if not isinstance(data, list):
            return
    except Exception as exc:
        logger.exception("failed to read %s: %s", jp, exc)
        if RAISE_ERRORS:
            raise HistoryParseError(str(exc)) from exc
        return
    conn = connect(db_path)
    for entry in data:
        if isinstance(entry, dict):
            insert_entry(conn, {str(k): float(v) for k, v in entry.items()})


def record(entry: Dict[str, float], path: str | Path | None = None) -> None:
    """Append ``entry`` to the history database."""
    conn = connect(path)
    insert_entry(conn, entry)


__all__ = [
    "connect",
    "connect_locked",
    "load_history",
    "insert_entry",
    "fetch_all",
    "fetch_after",
    "fetch_latest",
    "migrate_json_to_db",
    "record",
]
