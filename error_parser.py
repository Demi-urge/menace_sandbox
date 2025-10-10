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

try:
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for package layout
    from .dynamic_path_router import resolve_path  # type: ignore

try:
    from .self_improvement.target_region import (
        TargetRegion,
        extract_target_region as _extract_target_region,
    )
except Exception:  # pragma: no cover - fallback for direct execution
    try:
        from import_compat import load_internal

        module = load_internal("self_improvement.target_region")
        TargetRegion = module.TargetRegion  # type: ignore[attr-defined]
        _extract_target_region = module.extract_target_region  # type: ignore[attr-defined]
    except Exception:
        from dataclasses import dataclass as _dataclass

        @_dataclass
        class _TargetRegionFallback:  # pragma: no cover - minimal stub
            module: str | None = None
            function: str | None = None
            start_line: int | None = None
            end_line: int | None = None

        TargetRegion = _TargetRegionFallback  # type: ignore

        def _extract_target_region(trace: str):  # type: ignore
            return None


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

    _cache = FailureCache()

    @staticmethod
    def parse_failure(output: str) -> dict[str, Optional[str]]:
        report = parse_failure(output)
        first_tag = report.tags[0] if report.tags else ""

        file: Optional[str] = None
        line_no: Optional[str] = None
        function: Optional[str] = None

        for frame_line in reversed(report.trace.splitlines()):
            frame_line = frame_line.strip()
            m = re.match(r'File "([^"]+)", line (\d+), in (.+)', frame_line)
            if m:
                file, line_no, function = m.group(1), m.group(2), m.group(3)
                break
            m = re.match(r'([^:\s]+\.py):(\d+): in (.+)', frame_line)
            if m:
                file, line_no, function = m.group(1), m.group(2), m.group(3)
                break

        return {
            "exception": first_tag,
            "file": file,
            "line": line_no,
            "function": function,
            "context": "",
            "strategy_tag": first_tag,
            "signature": _signature(report.trace),
            "timestamp": datetime.utcnow().isoformat(),
            "stack": report.trace,
        }

    @staticmethod
    def parse(trace: str) -> dict:
        """Parse ``trace`` into a structured dictionary.

        The result contains canonical ``tags`` derived from :func:`parse_failure`,
        the primary ``error_type`` (first tag), a list of referenced ``files`` and
        a ``signature`` uniquely identifying the trace.  Duplicate traces are
        skipped using :class:`FailureCache` and return an empty dictionary.
        """

        report = parse_failure(trace)
        if ErrorParser._cache.seen(report):
            return {}
        ErrorParser._cache.add(report)

        files: list[str] = []
        for line in report.trace.splitlines():
            m = re.match(r'File "([^"]+)", line \d+', line)
            if m:
                files.append(m.group(1))
                continue
            m = re.match(r'([^:\s]+\.py):\d+:', line)
            if m:
                files.append(m.group(1))
        # remove duplicates while preserving order
        seen_files: dict[str, None] = {}
        for f in files:
            seen_files.setdefault(f, None)

        region = _extract_target_region(report.trace)
        first_tag = report.tags[0] if report.tags else ""
        return {
            "error_type": first_tag,
            "files": list(seen_files.keys()),
            "tags": report.tags,
            "signature": _signature(report.trace),
            "trace": report.trace,
            "target_region": region,
        }


def extract_target_region(trace: str) -> TargetRegion | None:
    """Expose targeting helper without importing ``self_improvement`` package."""

    return _extract_target_region(trace)


__all__ = [
    "ErrorReport",
    "parse_failure",
    "FailureCache",
    "ErrorParser",
    "TargetRegion",
    "extract_target_region",
]
