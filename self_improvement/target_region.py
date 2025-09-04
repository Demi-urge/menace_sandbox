"""Unified representation of a source code region.

This module exposes :class:`TargetRegion` along with helper functions to
derive regions from stack traces.  The dataclass is intentionally lightweight
so that it can be passed between different components without pulling in heavy
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import re


@dataclass
class TargetRegion:
    """Contiguous region within a source file.

    Parameters
    ----------
    start_line:
        First line number of the region (1-indexed).
    end_line:
        Last line number of the region (1-indexed).
    function:
        Name of the enclosing function or ``"<module>"`` when at module level.
    filename:
        Path to the file containing the region.  Defaults to an empty string
        when unknown so the class can be used in contexts where only line
        information is available.
    """

    start_line: int
    end_line: int
    function: str
    filename: str = ""


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')


def region_from_frame(filename: str, lineno: int, func_name: str) -> TargetRegion:
    """Return a :class:`TargetRegion` for ``filename``/``lineno``.

    The source file is parsed with :mod:`ast` to determine the smallest
    function or class body enclosing ``lineno``.  If parsing fails a region
    spanning only ``lineno`` is returned.
    """

    path = Path(filename)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return TargetRegion(start_line=lineno, end_line=lineno, function=func_name, filename=str(path))

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
        name = getattr(target, "name", func_name)
        return TargetRegion(start_line=start, end_line=end, function=name, filename=str(path))

    return TargetRegion(start_line=lineno, end_line=lineno, function=func_name, filename=str(path))


def extract_target_region(trace: str) -> TargetRegion | None:
    """Extract the innermost relevant frame from ``trace``.

    Only frames that reside inside the repository root are considered.  The
    deepest such frame is converted into a :class:`TargetRegion`.  ``None`` is
    returned when no matching frame is found.
    """

    frames = _FRAME_RE.findall(trace)
    if not frames:
        return None

    repo_root = Path(__file__).resolve().parent.parent
    for filename, lineno_str, func in reversed(frames):
        try:
            path = Path(filename).resolve()
        except Exception:
            continue
        try:
            path.relative_to(repo_root)
        except ValueError:
            continue
        return region_from_frame(str(path), int(lineno_str), func.strip())
    return None


__all__ = ["TargetRegion", "region_from_frame", "extract_target_region"]

