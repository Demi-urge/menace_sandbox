"""Utilities for persisting synergy history.

The module relies on the global database router.  When executed as a script it
initialises the router with the identifier ``"synergy_history_db"``.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Literal
from contextlib import contextmanager
import logging
from filelock import FileLock

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from . import RAISE_ERRORS

try:  # pragma: no cover - package context
    from .scope_utils import Scope, build_scope_clause, apply_scope
except Exception:  # pragma: no cover - fallback for script usage
    from scope_utils import Scope, build_scope_clause, apply_scope  # type: ignore

router = GLOBAL_ROUTER or init_db_router("synergy_history_db")

logger = logging.getLogger(__name__)


class HistoryParseError(RuntimeError):
    """Raised when synergy history parsing fails."""


CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS synergy_history ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " source_menace_id TEXT NOT NULL,"
    " entry TEXT NOT NULL)"
)


def connect(
    path: str | Path | None = None,
    *,
    router: DBRouter | None = None,
) -> sqlite3.Connection:
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
    cols = [
        row[1]
        for row in conn.execute("PRAGMA table_info(synergy_history)").fetchall()
    ]
    if "source_menace_id" not in cols:
        conn.execute(
            "ALTER TABLE synergy_history ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            "UPDATE synergy_history SET source_menace_id='' WHERE source_menace_id IS NULL"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_synergy_history_source_menace_id "
        "ON synergy_history(source_menace_id)"
    )
    conn.commit()
    return conn


@contextmanager
def connect_locked(
    path: str | Path | None = None,
    *,
    router: DBRouter | None = None,
) -> Iterator[sqlite3.Connection]:
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


def load_history(
    path: str | Path | None = None,
    *,
    router: DBRouter | None = None,
) -> List[Dict[str, float]]:
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


def insert_entry(
    conn: sqlite3.Connection, entry: Dict[str, float], *, source_menace_id: str | None = None
) -> None:
    menace_id = source_menace_id or getattr(globals().get("router"), "menace_id", "")
    path = None
    try:
        path = conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:
        pass
    sql = "INSERT INTO synergy_history(source_menace_id, entry) VALUES (?, ?)"
    params = (menace_id, json.dumps(entry))
    if path:
        lock = FileLock(str(path) + ".lock")
        with lock:
            conn.execute(sql, params)
            conn.commit()
    else:
        conn.execute(sql, params)
        conn.commit()


def fetch_all(
    conn: sqlite3.Connection,
    *,
    scope: Literal["local", "global", "all"] = "local",
    source_menace_id: str | None = None,
) -> List[Dict[str, float]]:
    menace_id = source_menace_id or getattr(globals().get("router"), "menace_id", "")
    clause, params = build_scope_clause("synergy_history", Scope(scope), menace_id)
    query = apply_scope("SELECT entry FROM synergy_history", clause) + " ORDER BY id"
    rows = conn.execute(query, params).fetchall()
    out: List[Dict[str, float]] = []
    for (text,) in rows:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                out.append({str(k): float(v) for k, v in data.items()})
        except Exception:
            continue
    return out


def fetch_after(
    conn: sqlite3.Connection,
    last_id: int,
    *,
    scope: Literal["local", "global", "all"] = "local",
    source_menace_id: str | None = None,
) -> List[Tuple[int, Dict[str, float]]]:
    """Return entries with ``id`` greater than ``last_id``."""
    menace_id = source_menace_id or getattr(globals().get("router"), "menace_id", "")
    clause, scope_params = build_scope_clause("synergy_history", Scope(scope), menace_id)
    query = apply_scope(
        "SELECT id, entry FROM synergy_history WHERE id > ?", clause
    ) + " ORDER BY id"
    params = [int(last_id), *scope_params]
    rows = conn.execute(query, params).fetchall()
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


def fetch_latest(
    conn: sqlite3.Connection,
    *,
    scope: Literal["local", "global", "all"] = "local",
    source_menace_id: str | None = None,
) -> Dict[str, float]:
    menace_id = source_menace_id or getattr(globals().get("router"), "menace_id", "")
    clause, params = build_scope_clause("synergy_history", Scope(scope), menace_id)
    query = apply_scope("SELECT entry FROM synergy_history", clause) + " ORDER BY id DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
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


def record(
    entry: Dict[str, float],
    path: str | Path | None = None,
    *,
    source_menace_id: str | None = None,
) -> None:
    """Append ``entry`` to the history database."""
    conn = connect(path)
    insert_entry(conn, entry, source_menace_id=source_menace_id)


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
