from __future__ import annotations

"""Utilities for locating failing regions from stack traces."""

from pathlib import Path
import ast
import re
import traceback
from types import TracebackType
from typing import Optional

try:
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore

try:
    from .self_improvement.target_region import TargetRegion
except Exception:  # pragma: no cover - fallback for direct execution
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_target_region_fallback",
        resolve_path("self_improvement/target_region.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_target_region_fallback"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    TargetRegion = module.TargetRegion  # type: ignore


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
