#!/usr/bin/env python3
"""Audit bot classes for missing ``@self_coding_managed`` decorators.

Walks the repository to find classes whose names end with ``Bot`` or
include methods named ``generate`` or ``refactor``. Any such class lacking
``@self_coding_managed`` is reported.

Usage:
    python self_coding_audit.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple

DECORATOR = "self_coding_managed"
METHOD_NAMES = {"generate", "refactor"}


def _has_decorator(node: ast.ClassDef) -> bool:
    """Return True if ``node`` uses ``@self_coding_managed``."""
    for deco in node.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == DECORATOR:
            return True
        if isinstance(deco, ast.Attribute) and deco.attr == DECORATOR:
            return True
    return False


def _is_relevant_class(node: ast.ClassDef) -> bool:
    """Check if class name ends with ``Bot`` or defines target methods."""
    if node.name.endswith("Bot"):
        return True
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name in METHOD_NAMES:
            return True
    return False


def _iter_python_files(root: Path) -> Iterable[Path]:
    """Yield all ``.py`` files under ``root`` except this script."""
    for path in root.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        yield path


def find_unmanaged_bots(root: Path) -> List[Tuple[Path, str, int]]:
    """Return list of unmanaged bot classes as ``(path, name, lineno)``."""
    unmanaged: List[Tuple[Path, str, int]] = []
    for path in _iter_python_files(root):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except Exception:
            # Skip files we cannot parse
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_relevant_class(node):
                if not _has_decorator(node):
                    unmanaged.append((path, node.name, node.lineno))
    return unmanaged


def main() -> None:
    root = Path(__file__).resolve().parent
    unmanaged = find_unmanaged_bots(root)
    if unmanaged:
        print("Found unmanaged bot classes:")
        for path, name, lineno in unmanaged:
            rel_path = path.relative_to(root)
            print(f" - {rel_path}:{lineno} -> {name} missing @self_coding_managed")
        print("Consider applying @self_coding_managed to the classes above.")
    else:
        print("All bot classes are managed.")


if __name__ == "__main__":
    main()
