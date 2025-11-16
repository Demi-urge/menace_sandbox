#!/usr/bin/env python3
"""Scan all subpackages for bot classes missing ``@self_coding_managed``."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, Sequence

# Modules where ``self_coding_managed`` is intentionally absent despite
# calling helper functions.
EXCLUDED_PATHS = {Path("self_coding_engine.py")}

# Helper invocation names that require @self_coding_managed.
HELPER_NAMES = {"manager_generate_helper", "generate_helper"}


def _calls_helper(cls: ast.ClassDef) -> bool:
    """Return True if *cls* calls one of the helper functions."""
    for node in ast.walk(cls):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in HELPER_NAMES:
                return True
    return False


def _has_decorator(cls: ast.ClassDef) -> bool:
    """Return True if *cls* has the self_coding_managed decorator."""
    for dec in cls.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "self_coding_managed":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "self_coding_managed":
            return True
    return False


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for base in paths:
        base = base.resolve()
        if base.is_dir():
            candidates = base.rglob("*.py")
        else:
            candidates = [base]
        for path in candidates:
            if "tests" in path.parts or "unit_tests" in path.parts:
                continue
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    paths = [root] if not argv else [root / a for a in argv]
    offenders: list[tuple[Path, str]] = []
    for path in _iter_python_files(paths):
        rel = path.relative_to(root)
        if rel in EXCLUDED_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                needs_check = (
                    node.name.endswith("Bot") and not node.name.startswith("_")
                ) or _calls_helper(node)
                if needs_check and not _has_decorator(node):
                    offenders.append((rel, node.name))
    if offenders:
        for path, cls in offenders:
            print(f"{path}:{cls} missing @self_coding_managed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
