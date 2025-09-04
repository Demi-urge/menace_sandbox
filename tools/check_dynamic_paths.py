#!/usr/bin/env python3
"""Pre-commit check for dynamic path references.

The hook scans Python source files for string literals that indicate a
repository path such as those containing ``sandbox_data`` or strings with a
``/`` and ending in ``.py``. If any such string is present but the file lacks a
``resolve_path`` call, the check fails to encourage portable path resolution.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable

TRIGGER_SUBSTRINGS = ["sandbox_data", "logs"]
SCRIPT_NAME = Path(__file__).name

# File extensions that should be treated as potential path references. ``.py``
# continues to require a ``/`` in the string to avoid false positives on module
# names, while the other extensions are flagged regardless of the presence of a
# path separator so that bare filenames like ``sandbox_settings.yaml`` are
# caught.
_EXTENSIONS_NEED_SLASH = (".py",)
_EXTENSIONS_ALWAYS = (".yaml", ".yml", ".json", ".jsonl", ".db", ".log")


def _triggers(value: str) -> bool:
    if any(sub in value for sub in TRIGGER_SUBSTRINGS):
        return True
    if any(value.endswith(ext) for ext in _EXTENSIONS_ALWAYS):
        return True
    if "/" in value and any(value.endswith(ext) for ext in _EXTENSIONS_NEED_SLASH):
        return True
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_resolve_path = False
        self.found_trigger = False

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name == "resolve_path":
            self.has_resolve_path = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # type: ignore[override]
        if isinstance(node.value, str) and _triggers(node.value):
            self.found_trigger = True


def check_file(path: str) -> bool:
    if Path(path).name == SCRIPT_NAME:
        return True
    with open(path, "r", encoding="utf-8") as handle:
        try:
            tree = ast.parse(handle.read(), filename=path)
        except SyntaxError:
            return True
    visitor = _Visitor()
    visitor.visit(tree)
    if visitor.found_trigger and not visitor.has_resolve_path:
        print(f"{path}: missing resolve_path for dynamic path references")
        return False
    return True


def main(argv: Iterable[str]) -> int:
    ok = True
    for file_path in argv:
        if not check_file(file_path):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
