"""Utilities for describing regions of source code.

This module provides :class:`TargetRegion`, a small dataclass that identifies a
contiguous region within a file along with helper functions used throughout the
system to locate such regions from stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import re
import traceback
from types import TracebackType
from typing import Optional


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')


@dataclass(init=False)
class TargetRegion:
    """Contiguous region in a source file.

    Parameters mirror historic variants with generous keyword aliases for
    backwards compatibility.  ``file``/``path``/``filename`` refer to the file
    containing the region and ``function``/``func_name`` to the enclosing
    function name.
    """

    start_line: int
    end_line: int
    function: str
    filename: str

    def __init__(
        self,
        start_line: int,
        end_line: int,
        function: str | None = None,
        filename: str | None = None,
        **kwargs: str,
    ) -> None:
        if function is None:
            function = kwargs.pop("func_name", "")
        else:
            kwargs.pop("func_name", None)
        if filename is None:
            filename = kwargs.pop("file", kwargs.pop("path", ""))
        else:
            kwargs.pop("file", None)
            kwargs.pop("path", None)
        if kwargs:
            raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
        self.start_line = start_line
        self.end_line = end_line
        self.function = function or ""
        self.filename = filename or ""

    # Backwards compatible attribute aliases ---------------------------------
    @property
    def file(self) -> str:  # pragma: no cover - backwards compat
        return self.filename

    @file.setter
    def file(self, value: str) -> None:  # pragma: no cover - backwards compat
        self.filename = value

    @property
    def path(self) -> str:  # pragma: no cover - backwards compat
        return self.filename

    @path.setter
    def path(self, value: str) -> None:  # pragma: no cover - backwards compat
        self.filename = value

    @property
    def func_name(self) -> str:  # pragma: no cover - backwards compat
        return self.function

    @func_name.setter
    def func_name(self, value: str) -> None:  # pragma: no cover - backwards compat
        self.function = value


# Helper functions -----------------------------------------------------------

def region_from_frame(filename: str, lineno: int, func: str) -> TargetRegion:
    """Return :class:`TargetRegion` for ``filename``/``lineno``.

    The file is parsed with :mod:`ast` to determine the smallest function or
    class body enclosing ``lineno``.  When parsing fails the region spans only
    the provided line.
    """

    path = Path(filename)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return TargetRegion(lineno, lineno, func, str(path))

    target: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is None or end is None:
                continue
            if start <= lineno <= end:
                if target is None or start >= getattr(target, "lineno", 0):
                    target = node
    if target is not None:
        start = getattr(target, "lineno", lineno)
        end = getattr(target, "end_lineno", start)
        name = getattr(target, "name", func)
        return TargetRegion(start, end, name, str(path))

    return TargetRegion(lineno, lineno, func, str(path))


def extract_target_region(trace: str | TracebackType) -> Optional[TargetRegion]:
    """Extract the innermost relevant frame from ``trace``."""

    tb: TracebackType | None = trace if isinstance(trace, TracebackType) else None
    if tb is not None:
        frames = list(traceback.walk_tb(tb))
        if not frames:
            return None
        frame, lineno = frames[-1]
        filename = frame.f_code.co_filename
        func = frame.f_code.co_name
        return region_from_frame(filename, lineno, func)

    frames = _FRAME_RE.findall(trace)
    if not frames:
        return None
    filename, lineno, func = frames[-1]
    return region_from_frame(filename, int(lineno), func.strip())


__all__ = ["TargetRegion", "region_from_frame", "extract_target_region"]

