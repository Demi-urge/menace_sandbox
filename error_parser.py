"""Utilities for parsing and storing structured error information.

This module exposes :class:`ErrorParser` with a :meth:`parse_failure` helper
that extracts key details from a raw traceback string.  The extracted data is
persisted to ``errors.db`` so that other components can analyse recurring
failures.

The parser returns a dictionary with the following keys:

``exception``
    Name of the raised exception, e.g. ``ValueError``.
``file`` / ``line``
    Location of the innermost stack frame.
``context``
    Source line from the failing file when available.
``strategy_tag``
    Deterministic tag derived from the failure which can be used for grouping
    similar issues.
``signature``
    SHA1 hash of the extracted information.
``timestamp``
    ISO formatted time when the record was stored.
``stack``
    Original log or stack trace passed to the parser.

The data is stored in an ``errors.db`` SQLite database using a very small table
named ``parsed_failures``.  The table is created on first use.
"""

from __future__ import annotations

from dataclasses import dataclass  # kept for FailureCache convenience
from datetime import datetime
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
except Exception:  # pragma: no cover - fallback when db_router unavailable
    import sqlite3  # type: ignore

    class DBRouter:  # type: ignore[override]
        def __init__(self, path: str) -> None:
            self.path = path

        def get_connection(self, name: str):  # noqa: ANN001 - simple shim
            return getattr(sqlite3, "connect")(self.path)

    GLOBAL_ROUTER = None  # type: ignore

    def init_db_router(name: str, local: str, shared: str) -> DBRouter:  # noqa: ARG001
        return DBRouter(local)


class ErrorParser:
    """Parse traceback strings for quick error analysis."""

    # ``File "...", line 123`` from Python tracebacks
    _TRACE_RE = re.compile(r'File "(?P<file>[^"\n]+)", line (?P<line>\d+)')

    # final ``ValueError: message`` style line
    _EXC_RE = re.compile(r'(?P<exc>\w+(?:Error|Exception))(?::|\s|$)')

    _DB_PATH = Path("errors.db")
    _conn = None

    @classmethod
    def _get_conn(cls):
        if cls._conn is None:
            p = cls._DB_PATH.resolve()
            router = GLOBAL_ROUTER or init_db_router("errors_db", str(p), str(p))
            cls._conn = router.get_connection("errors")
            cls._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parsed_failures(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file TEXT,
                    line INTEGER,
                    context TEXT,
                    exception TEXT,
                    strategy_tag TEXT,
                    ts TEXT
                )
                """
            )
            cls._conn.commit()
        return cls._conn

    @staticmethod
    def _strategy_tag(exc: str, file: str, line: int) -> str:
        src = f"{exc}:{file}:{line}"
        return hashlib.sha1(src.encode("utf-8")).hexdigest()[:8]

    @classmethod
    def parse_failure(cls, log: str) -> Dict[str, Any]:
        """Return structured information extracted from ``log``.

        The resulting dictionary is also stored in ``errors.db``.
        """

        file: Optional[str] = None
        line_no: Optional[int] = None
        for m in cls._TRACE_RE.finditer(log):
            file = m.group("file")
            line_no = int(m.group("line"))

        context = ""
        if file and line_no is not None:
            try:
                src = Path(file).read_text(encoding="utf-8").splitlines()
                context = src[line_no - 1].strip()
            except Exception:  # pragma: no cover - best effort
                context = ""

        exc: Optional[str] = None
        for ln in reversed(log.splitlines()):
            m = cls._EXC_RE.search(ln.strip())
            if m:
                exc = m.group("exc")
                break

        strategy = cls._strategy_tag(exc or "", file or "", line_no or 0)
        signature = hashlib.sha1(
            f"{exc}:{file}:{line_no}:{context}".encode("utf-8")
        ).hexdigest()
        ts = datetime.utcnow().isoformat()

        conn = cls._get_conn()
        conn.execute(
            "INSERT INTO parsed_failures(file,line,context,exception,strategy_tag,ts)"
            " VALUES(?,?,?,?,?,?)",
            (file, line_no, context, exc, strategy, ts),
        )
        conn.commit()

        return {
            "exception": exc,
            "file": file,
            "line": line_no,
            "context": context,
            "strategy_tag": strategy,
            "signature": signature,
            "timestamp": ts,
            "stack": log,
        }


def parse_failure(log: str) -> Dict[str, Any]:
    """Module level convenience wrapper around :class:`ErrorParser`."""

    return ErrorParser.parse_failure(log)


class FailureCache:
    """Tiny SQLite backed cache for parsed failures.

    This is used by :mod:`self_debugger_sandbox` to avoid repeated processing
    of the same failure signature.  The cache itself is separate from the main
    ``errors.db`` telemetry and defaults to ``failures.db``.
    """

    def __init__(
        self,
        path: str | Path = "failures.db",
        *,
        router: DBRouter | None = None,
    ) -> None:
        p = Path(path).resolve()
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "parsed_failures_db", str(p), str(p)
        )
        self.conn = self.router.get_connection("parsed_failures")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS parsed_failures(""" "signature TEXT PRIMARY KEY,"
            " stack TEXT)"
        )
        self.conn.commit()

    def seen(self, signature: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM parsed_failures WHERE signature=?", (signature,)
        )
        return cur.fetchone() is not None

    def add(self, failure: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO parsed_failures(signature, stack) VALUES(?,?)",
            (failure["signature"], failure["stack"]),
        )
        self.conn.commit()


__all__ = ["ErrorParser", "parse_failure", "FailureCache"]

