from __future__ import annotations

"""Utilities for locating failing regions from stack traces."""

from dataclasses import dataclass
from pathlib import Path
import ast
import re
import traceback
from types import TracebackType
from typing import Optional


@dataclass
class TargetRegion:
    """Represents a region of code implicated by a failure."""

    path: str
    start_line: int
    end_line: int
    func_name: str


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')


def _region_from_path(path: str, line: int, func: str) -> TargetRegion:
    """Derive ``TargetRegion`` for ``path``/``line`` using AST introspection."""

    try:
        source = Path(path).read_text(encoding="utf-8")
    except Exception:
        return TargetRegion(path=path, start_line=line, end_line=line, func_name=func)

    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.lineno <= line <= getattr(node, "end_lineno", node.lineno):
                    end = getattr(node, "end_lineno", node.lineno)
                    name = getattr(node, "name", func)
                    return TargetRegion(
                        path=path,
                        start_line=node.lineno,
                        end_line=end,
                        func_name=name,
                    )
    except Exception:
        pass

    return TargetRegion(path=path, start_line=line, end_line=line, func_name=func)


def extract_target_region(trace: str | TracebackType) -> Optional[TargetRegion]:
    """Extract the innermost failing region from ``trace``.

    ``trace`` may be either a traceback string or a ``TracebackType``.  When a
    traceback object is provided, :func:`traceback.walk_tb` is used to locate the
    deepest frame.  Otherwise a best-effort regex parse of the string is
    performed.  ``None`` is returned if no usable frame information can be
    determined.
    """

    tb: TracebackType | None = trace if isinstance(trace, TracebackType) else None
    if tb is not None:
        frames = list(traceback.walk_tb(tb))
        if not frames:
            return None
        frame, lineno = frames[-1]
        path = frame.f_code.co_filename
        func = frame.f_code.co_name
        return _region_from_path(path, lineno, func)

    match = _FRAME_RE.findall(trace)
    if not match:
        return None
    path, line, func = match[-1]
    return _region_from_path(path, int(line), func)


__all__ = ["TargetRegion", "extract_target_region"]
