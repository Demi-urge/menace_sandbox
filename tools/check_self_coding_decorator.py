#!/usr/bin/env python3
"""Ensure classes using self-coding helpers are decorated."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

# Modules where ``self_coding_managed`` is intentionally absent despite
# calling ``manager_generate_helper``.
EXCLUDED_PATHS = {Path("self_coding_engine.py")}


def _calls_helper(cls: ast.ClassDef) -> bool:
    """Return True if *cls* calls manager_generate_helper or generate_helper."""
    for node in ast.walk(cls):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"manager_generate_helper", "generate_helper"}:
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


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        yield path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders: list[tuple[Path, str]] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(root)
        if rel in EXCLUDED_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            continue
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and _calls_helper(node):
                if not _has_decorator(node):
                    offenders.append((rel, node.name))
    if offenders:
        for path, cls in offenders:
            print(f"{path}:{cls} missing @self_coding_managed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
