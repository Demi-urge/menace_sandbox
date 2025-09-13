#!/usr/bin/env python3
"""Check that ``SelfCodingManager`` usage is properly registered.

The script scans all Python modules in the repository looking for
instantiations of :class:`SelfCodingManager`.  Any module (excluding test
modules and ``self_coding_manager.py`` itself) that constructs a manager must
either call ``internalize_coding_bot`` or decorate a class with
``@self_coding_managed``.  Offending modules are printed and the script exits
with a non-zero status so the check can be enforced in pre-commit/CI.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _instantiates_manager(tree: ast.AST) -> bool:
    """Return ``True`` if the AST contains ``SelfCodingManager(...)``."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "SelfCodingManager":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "SelfCodingManager":
                return True
    return False


def _calls_internalize(tree: ast.AST) -> bool:
    """Return ``True`` if ``internalize_coding_bot`` is called in the module."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "internalize_coding_bot":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "internalize_coding_bot":
                return True
    return False


def _has_managed_class(tree: ast.AST) -> bool:
    """Return ``True`` if any class uses the ``self_coding_managed`` decorator."""

    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    dec = dec.func
                if isinstance(dec, ast.Name) and dec.id == "self_coding_managed":
                    return True
                if isinstance(dec, ast.Attribute) and dec.attr == "self_coding_managed":
                    return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders: list[Path] = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        if path.name == "self_coding_manager.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _instantiates_manager(tree):
            continue
        if _calls_internalize(tree) or _has_managed_class(tree):
            continue
        offenders.append(path.relative_to(root))
    if offenders:
        for p in offenders:
            print(
                f"{p}: SelfCodingManager instantiated without internalize_coding_bot or @self_coding_managed"
            )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
