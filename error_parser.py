"""Utilities for parsing Python failure logs.

This module provides a lightweight :func:`parse_failure` helper that extracts
useful information from test logs such as stack traces and the final error
type.  A tiny SQLite backed :class:`FailureCache` is also exposed so callers can
persist failure signatures and avoid repeatedly attempting identical
reproductions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
try:  # pragma: no cover - optional dependency
    from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
except Exception:  # pragma: no cover - fallback when db_router unavailable
    import sqlite3  # type: ignore

    class DBRouter:  # type: ignore
        def __init__(self, path: str) -> None:
            self.path = path

        def get_connection(self, name: str):  # noqa: ANN001 - simple shim
            return getattr(sqlite3, "connect")(self.path)

    GLOBAL_ROUTER = None  # type: ignore

    def init_db_router(name: str, local: str, shared: str) -> DBRouter:  # noqa: ARG001
        return DBRouter(local)


class ErrorParser:
    """Parse traceback strings for quick error analysis."""

    # Match ``File "..."`` entries from standard Python tracebacks or bare
    # ``path/to/file.py`` lines from pytest output.
    _FILE_RE = re.compile(r'File "([^"]+)"|^([\w./-]+\.py)', re.MULTILINE)

    # Match the final error type such as ``ValueError`` or ``AssertionError``.
    _ERROR_RE = re.compile(
        r'^\s*(?:E\s+)?(?P<error>\w+(?:Error|Exception|Failed))', re.MULTILINE
    )

    # Heuristic mapping from error types to suggested tags.
    _TAG_MAP = {
        "AssertionError": ["test", "assertion"],
        "Failed": ["test"],
        "SyntaxError": ["syntax"],
        "TypeError": ["runtime"],
        "ValueError": ["runtime"],
        "ZeroDivisionError": ["runtime", "math"],
        "ModuleNotFoundError": ["missing-dependency"],
        "ImportError": ["missing-dependency"],
        "RuntimeError": ["runtime"],
    }

    @classmethod
    def parse(cls, trace: str) -> Dict[str, object]:
        """Parse ``trace`` and return extracted information.

        The return value is ``{"files": list[str], "error_type": str | None,
        "tags": list[str]}``.
        """

        files: List[str] = []
        for match in cls._FILE_RE.finditer(trace):
            path = match.group(1) or match.group(2)
            if path and path not in files:
                files.append(path)

        error_type = None
        last_line = trace.splitlines()[-1] if trace.splitlines() else ""
        match = cls._ERROR_RE.search(last_line)
        if not match:
            match = cls._ERROR_RE.search(trace)
        if match:
            error_type = match.group("error")

        tags = cls._TAG_MAP.get(error_type, [])
        return {"files": files, "error_type": error_type, "tags": tags}


@dataclass
class ParsedFailure:
    """Structured representation of a failure log."""

    stack_trace: str
    error_type: str | None
    reproduction_steps: List[str]
    signature: str


def parse_failure(log: str) -> ParsedFailure:
    """Return :class:`ParsedFailure` extracted from ``log``.

    Parameters
    ----------
    log:
        Raw stderr or traceback output from a failing command.
    """

    frames: List[str] = []
    for m in ErrorParser._FILE_RE.finditer(log):
        path = m.group(1) or m.group(2)
        if path:
            line_match = re.search(r"line (\d+)", log[m.start():m.end() + 40])
            if line_match:
                frames.append(f"{path}:{line_match.group(1)}")
            else:
                frames.append(path)

    parsed = ErrorParser.parse(log)
    error_type = parsed.get("error_type") if isinstance(parsed, dict) else None
    reproduction_steps = frames[-1:]  # minimal reproduction from last frame
    stack_trace = "\n".join(frames)
    sig_src = (error_type or "") + stack_trace
    signature = hashlib.sha1(sig_src.encode("utf-8")).hexdigest()
    return ParsedFailure(stack_trace, error_type, reproduction_steps, signature)


class FailureCache:
    """Tiny SQLite backed cache for parsed failures."""

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
            "CREATE TABLE IF NOT EXISTS parsed_failures(" "signature TEXT PRIMARY KEY, "
            "error_type TEXT, stack TEXT)"
        )
        self.conn.commit()

    def seen(self, signature: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM parsed_failures WHERE signature=?", (signature,)
        )
        return cur.fetchone() is not None

    def add(self, failure: ParsedFailure) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO parsed_failures(signature, error_type, stack) VALUES(?,?,?)",
            (failure.signature, failure.error_type, failure.stack_trace),
        )
        self.conn.commit()


__all__ = ["ErrorParser", "ParsedFailure", "parse_failure", "FailureCache"]
