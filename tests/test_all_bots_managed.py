"""Ensure all bot classes are decorated with ``@self_coding_managed``.

The test scans the repository for modules whose filenames include ``"bot"``
and inspects each class ending with the ``Bot`` suffix.  Any class missing the
``@self_coding_managed`` decorator causes the test to fail.

If a failure occurs, add ``@self_coding_managed`` to the class or register the
bot via ``BotRegistry.register_bot`` and log evaluations to mark it as
managed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _has_self_coding_managed(cls: ast.ClassDef) -> bool:
    return any(
        (isinstance(dec, ast.Name) and dec.id == "self_coding_managed")
        or (isinstance(dec, ast.Attribute) and dec.attr == "self_coding_managed")
        for dec in cls.decorator_list
    )


def test_all_bots_are_managed() -> None:
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in root.rglob("*bot*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Bot"):
                if not _has_self_coding_managed(node):
                    offenders.append(f"{path.relative_to(root)}:{node.name}")
    if offenders:
        formatted = "\n".join(offenders)
        pytest.fail(
            "Unregistered bot classes detected (missing @self_coding_managed):\n"
            f"{formatted}\n"
            "Remediation: decorate each class with @self_coding_managed or register "
            "the bot via BotRegistry.register_bot and log evaluations."
        )
