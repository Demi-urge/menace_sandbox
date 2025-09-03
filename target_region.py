from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import re
from typing import Optional


@dataclass
class TargetRegion:
    """Represents a contiguous region of source code implicated in a failure."""

    filename: str
    start_line: int
    end_line: int
    function: str


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')


def _region_from_frame(filename: str, lineno: int, func: str) -> TargetRegion:
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
        return TargetRegion(filename=str(path), start_line=lineno, end_line=lineno, function=func)

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
        return TargetRegion(filename=str(path), start_line=start, end_line=end, function=name)

    return TargetRegion(filename=str(path), start_line=lineno, end_line=lineno, function=func)


def extract_target_region(trace: str) -> Optional[TargetRegion]:
    """Extract the innermost relevant frame from ``trace``.

    The deepest frame within the stack trace is analysed and converted into a
    :class:`TargetRegion` using :func:`_region_from_frame`.  ``None`` is returned
    when no frame information can be determined.
    """

    frames = _FRAME_RE.findall(trace)
    if not frames:
        return None
    filename, lineno, func = frames[-1]
    return _region_from_frame(filename, int(lineno), func.strip())


__all__ = ["TargetRegion", "extract_target_region"]
