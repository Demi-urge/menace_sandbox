from __future__ import annotations

"""Minimal failure parsing utilities.

This module exposes :func:`parse_failure` which converts raw test output into
an :class:`ErrorReport`.  A small :class:`FailureCache` is provided to avoid
re-processing the same trace multiple times.  A compatibility :class:`ErrorParser`
wrapper mimics the old dictionary based API used by some callers.
"""

from dataclasses import dataclass
import hashlib
import re
from datetime import datetime
from typing import Optional


@dataclass
class ErrorReport:
    """Lightweight structured representation of a failure."""

    trace: str
    tags: list[str]


_CANON_RE = re.compile(r"(?<!^)(?=[A-Z])")
_ERROR_RE = re.compile(r"(\w+(?:Error|Exception))")


def _canonical(name: str) -> str:
    return _CANON_RE.sub("_", name).lower()


def _signature(trace: str) -> str:
    return hashlib.sha1(trace.encode("utf-8")).hexdigest()


def parse_failure(output: str) -> ErrorReport:
    """Extract stack trace and canonical error tags from ``output``."""

    match = re.search(r"(Traceback.*)", output, re.DOTALL)
    trace = match.group(1) if match else output
    tags: list[str] = []
    seen: set[str] = set()
    for exc in _ERROR_RE.findall(output):
        tag = _canonical(exc)
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return ErrorReport(trace=trace, tags=tags)


class FailureCache:
    """Very small in-memory cache keyed by trace signature."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen(self, report: ErrorReport | str) -> bool:
        trace = report.trace if isinstance(report, ErrorReport) else report
        return _signature(trace) in self._seen

    def add(self, report: ErrorReport) -> None:
        self._seen.add(_signature(report.trace))


class ErrorParser:
    """Backward compatible dictionary interface."""

    @staticmethod
    def parse_failure(output: str) -> dict[str, Optional[str]]:
        report = parse_failure(output)
        first_tag = report.tags[0] if report.tags else ""
        return {
            "exception": first_tag,
            "file": None,
            "line": None,
            "context": "",
            "strategy_tag": first_tag,
            "signature": _signature(report.trace),
            "timestamp": datetime.utcnow().isoformat(),
            "stack": report.trace,
        }


__all__ = ["ErrorReport", "parse_failure", "FailureCache", "ErrorParser"]
