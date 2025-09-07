#!/usr/bin/env python3
"""Static check for ``ContextBuilder`` usage.

This script scans the repository for calls to ``_build_prompt``,
``PromptEngine`` and patch helpers such as ``generate_patch`` to ensure that a
``context_builder`` keyword argument is supplied.  It now also checks *any*
function call named ``build_prompt`` or ``build_prompt_with_memory`` regardless
of whether it is accessed as an attribute.  The check ignores any files located
in directories named ``tests`` or ``unit_tests``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_NAMES = {
    "PromptEngine",
    "_build_prompt",
    "generate_patch",
    "build_prompt",
    "build_prompt_with_memory",
}


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in {"tests", "unit_tests"} for part in path.parts):
            continue
        yield path


def check_file(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:  # pragma: no cover - syntax errors
        print(f"Skipping {path}: {exc}", file=sys.stderr)
        return []

    errors: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
            fn = node.func
            name: str | None = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            has_kw = any(kw.arg == "context_builder" for kw in node.keywords)

            if name in REQUIRED_NAMES and not has_kw:
                errors.append((node.lineno, name))

            self.generic_visit(node)

    Visitor().visit(tree)
    return errors


def main() -> int:
    failures: list[str] = []
    for path in iter_python_files(ROOT):
        for lineno, name in check_file(path):
            failures.append(f"{path}:{lineno} -> {name} missing context_builder")
    if failures:
        for line in failures:
            print(line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
