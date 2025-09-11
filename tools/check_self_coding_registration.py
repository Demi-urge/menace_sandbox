#!/usr/bin/env python3
"""Lint check ensuring all coding bots use ``@self_coding_managed``.

The script scans the repository for modules that import
``self_coding_manager`` or ``self_coding_engine`` and looks for classes
with names ending in ``Bot``.  Any such class must be decorated with the
``self_coding_managed`` decorator.  The check ignores test modules.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys


def _uses_self_coding(tree: ast.AST) -> bool:
    """Return ``True`` if *tree* imports self-coding helpers."""

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("self_coding_manager") or mod.endswith("self_coding_engine"):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("self_coding_manager") or alias.name.endswith(
                    "self_coding_engine"
                ):
                    return True
    return False


def _missing_decorator(tree: ast.AST) -> list[str]:
    """Return class names missing the ``self_coding_managed`` decorator."""

    missing: list[str] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name.endswith("Bot"):
            has_dec = any(
                (isinstance(dec, ast.Name) and dec.id == "self_coding_managed")
                or (isinstance(dec, ast.Attribute) and dec.attr == "self_coding_managed")
                for dec in node.decorator_list
            )
            if not has_dec:
                missing.append(node.name)
    return missing


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders: list[tuple[Path, list[str]]] = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _uses_self_coding(tree):
            continue
        missing = _missing_decorator(tree)
        if missing:
            offenders.append((path.relative_to(root), missing))
    if offenders:
        for path, classes in offenders:
            cls_list = ", ".join(classes)
            print(f"{path}: missing @self_coding_managed on {cls_list}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())

