#!/usr/bin/env python3
"""Audit bot classes for missing ``@self_coding_managed`` decorators.

Walks the repository to find classes whose names end with ``Bot``. Any such
class lacking ``@self_coding_managed`` is reported.

Usage:
    python self_coding_audit.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

DECORATOR = "self_coding_managed"
EXCLUDED_PATHS = {
    Path("config.py"),
    Path("investment_engine.py"),
    Path("revenue_amplifier.py"),
    Path("plugins/metrics_prediction.py"),
    Path("data_bot.py"),
    Path("database_manager.py"),
}


def _has_decorator(node: ast.ClassDef) -> bool:
    """Return True if ``node`` uses ``@self_coding_managed``."""
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Name) and target.id == DECORATOR:
            return True
        if isinstance(target, ast.Attribute) and target.attr == DECORATOR:
            return True
    return False


def _has_register_and_log(tree: ast.AST) -> bool:
    """Return True if module registers the bot and logs evaluations."""

    has_reg = False
    has_log = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "register_bot":
                has_reg = True
            elif node.func.attr == "log_eval":
                has_log = True
        if has_reg and has_log:
            return True
    return False


def _is_relevant_class(node: ast.ClassDef) -> bool:
    """Return True if class name ends with ``Bot``."""
    return node.name.endswith("Bot")


def _iter_python_files(root: Path) -> Iterable[Path]:
    """Yield all ``.py`` files under ``root`` except this script."""
    for path in root.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if "tests" in parts or "unit_tests" in parts:
            continue
        if rel in EXCLUDED_PATHS:
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
        has_reg_log = _has_register_and_log(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_relevant_class(node):
                if not _has_decorator(node) and not has_reg_log:
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
        sys.exit(1)
    else:
        print("All bot classes are managed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
