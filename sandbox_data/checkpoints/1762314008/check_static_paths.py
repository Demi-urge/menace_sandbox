#!/usr/bin/env python3
"""Pre-commit check for hard coded `.py` paths.

The hook scans Python source files for string literals ending with ``.py``.
Any such literal must be wrapped by :func:`resolve_path` or
:func:`path_for_prompt` to ensure paths remain portable across forks and
clones.  Strings starting with ``*`` (e.g. ``"*.py"``) or equal to
``"__init__.py"`` are ignored as they are typically wildcards or module
names.  A line can be explicitly skipped by adding the
inline comment ``# path-ignore``.
"""

from __future__ import annotations

import ast
import sys
from typing import Iterable, List, Set

ALLOWED_FUNCS = {"resolve_path", "path_for_prompt"}
PY_EXT = ".py"


def _is_py_string(node: ast.Constant) -> bool:
    if not (
        isinstance(node.value, str)
        and node.value.endswith(PY_EXT)
        and len(node.value) > len(PY_EXT)
    ):
        return False

    val = node.value

    if val.startswith("*"):
        return False

    if val == "__init__.py" and "/" not in val and "\\" not in val:
        return False

    return True


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.all_nodes: List[ast.Constant] = []
        self.allowed_nodes: Set[ast.Constant] = set()

    def visit_Constant(self, node: ast.Constant) -> None:  # type: ignore[override]
        if _is_py_string(node):
            self.all_nodes.append(node)

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name in ALLOWED_FUNCS:
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Constant) and _is_py_string(sub_node):
                    self.allowed_nodes.add(sub_node)
        self.generic_visit(node)


def check_file(path: str) -> List[ast.Constant]:
    with open(path, "r", encoding="utf-8") as handle:
        try:
            content = handle.read()
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            return []
    lines = content.splitlines()
    collector = _Collector()
    collector.visit(tree)
    return [
        n
        for n in collector.all_nodes
        if n not in collector.allowed_nodes
        and "# path-ignore" not in lines[n.lineno - 1]
    ]


def main(argv: Iterable[str]) -> int:
    has_error = False
    for file_path in argv:
        for node in check_file(file_path):
            print(
                f"{file_path}:{node.lineno}: static path '{node.value}' not wrapped "
                "with resolve_path or path_for_prompt",
            )
            has_error = True
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
