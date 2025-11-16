"""Flake8 plugin warning against direct ``sqlite3.connect`` usage."""
from __future__ import annotations

import ast
from typing import Generator, Tuple, Type


class Plugin:
    """Simple AST checker used as a flake8 plugin.

    It emits ``SQL001`` whenever a call to ``sqlite3.connect`` is encountered.
    The rule is advisory and mirrors the runtime test guarding against direct
    SQLite connections.
    """

    name = "flake8-sqlite-connect"
    version = "0.1.0"

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def run(self) -> Generator[Tuple[int, int, str, Type[object]], None, None]:
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sqlite3"
                and node.func.attr == "connect"
            ):
                msg = "SQL001 avoid direct sqlite3.connect calls; use DBRouter instead"
                yield node.lineno, node.col_offset, msg, type(self)
