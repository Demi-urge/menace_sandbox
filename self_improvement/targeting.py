from __future__ import annotations

from dataclasses import dataclass
import ast
import pathlib
import re
# ``typing`` is intentionally minimal to keep this module lightweight.


@dataclass
class TargetRegion:
    """Represents a contiguous region of source code.

    Attributes
    ----------
    file: str
        Absolute path to the source file containing the target.
    start_line: int
        First line number of the region (1-indexed).
    end_line: int
        Last line number of the region (1-indexed).
    func_name: str
        Name of the enclosing function or ``"<module>"`` when the code is at
        module level.
    """

    file: str
    start_line: int
    end_line: int
    func_name: str


def region_from_frame(filename: str, lineno: int, func_name: str) -> TargetRegion:
    """Return :class:`TargetRegion` for ``filename``/``lineno``.

    The module is parsed using :mod:`ast` and the smallest function (or class)
    enclosing ``lineno`` is returned.  If parsing fails or no suitable node is
    found, a region spanning only ``lineno`` is returned.
    """

    path = pathlib.Path(filename)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return TargetRegion(file=str(path), start_line=lineno, end_line=lineno, func_name=func_name)

    target: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start is None or end is None:
                continue
            if start <= lineno <= end:
                # choose the innermost function/class containing the line
                if target is None or start >= getattr(target, "lineno", 0):
                    target = node
    if target is not None:
        start = getattr(target, "lineno", lineno)
        end = getattr(target, "end_lineno", start)
        name = getattr(target, "name", func_name)
        return TargetRegion(file=str(path), start_line=start, end_line=end, func_name=name)

    return TargetRegion(file=str(path), start_line=lineno, end_line=lineno, func_name=func_name)


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')


def extract_target_region(trace: str) -> TargetRegion | None:
    """Parse ``trace`` and return the innermost frame within the repo.

    The stack trace is scanned for ``File ...`` entries.  The deepest frame whose
    filename resides inside the repository root is returned as a
    :class:`TargetRegion`.  ``None`` is returned when no such frame is found.
    """

    frames = _FRAME_RE.findall(trace)
    if not frames:
        return None

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for filename, lineno_str, func in reversed(frames):
        try:
            path = pathlib.Path(filename).resolve()
        except Exception:
            continue
        try:
            path.relative_to(repo_root)
        except ValueError:
            continue
        return region_from_frame(str(path), int(lineno_str), func.strip())
    return None


__all__ = ["TargetRegion", "region_from_frame", "extract_target_region"]
