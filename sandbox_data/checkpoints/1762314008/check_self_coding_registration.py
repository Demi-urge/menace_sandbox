#!/usr/bin/env python3
"""Check that coding bots are properly registered.

The script scans all Python modules in the repository looking for either
instantiations of :class:`SelfCodingManager` *or* modules that export classes or
functions whose names end with ``Bot``.  Any such module (excluding test modules
and ``self_coding_manager.py`` itself) must either call
``internalize_coding_bot`` or decorate the bot with ``@self_coding_managed``.
Offending modules are printed and the script exits with a non-zero status so
the check can be enforced in pre-commit/CI.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _has_self_coding_decorator(node: ast.AST) -> bool:
    """Return ``True`` if ``node`` has the ``self_coding_managed`` decorator."""

    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Call):
            dec = dec.func
        if isinstance(dec, ast.Name) and dec.id == "self_coding_managed":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "self_coding_managed":
            return True
    return False


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


def _has_managed_entity(tree: ast.AST) -> bool:
    """Return ``True`` if any top-level class or function uses the decorator."""

    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if _has_self_coding_decorator(node):
                return True
    return False


def _exported_bots(tree: ast.AST) -> list[ast.AST]:
    """Return top-level bot definitions exported by the module."""

    bots = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.endswith("Bot") and not node.name.startswith("_"):
                bots.append(node)
    return bots


def _unmanaged_exports(tree: ast.AST, calls_internalize: bool) -> list[str]:
    """Return names of exported bots lacking registration or decoration."""

    if calls_internalize:
        return []
    return [node.name for node in _exported_bots(tree) if not _has_self_coding_decorator(node)]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manager_offenders: list[Path] = []
    export_offenders: list[tuple[Path, list[str]]] = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        if path.name == "self_coding_manager.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        calls_internalize = _calls_internalize(tree)
        unmanaged = _unmanaged_exports(tree, calls_internalize)
        has_manager = _instantiates_manager(tree)
        if has_manager and not (calls_internalize or _has_managed_entity(tree)):
            manager_offenders.append(path.relative_to(root))
        if unmanaged:
            export_offenders.append((path.relative_to(root), unmanaged))
    if manager_offenders or export_offenders:
        for p in manager_offenders:
            print(
                f"{p}: SelfCodingManager instantiated without "
                "internalize_coding_bot or @self_coding_managed",
            )
        for p, bots in export_offenders:
            print(f"{p}: unmanaged exported bots: {', '.join(bots)}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
