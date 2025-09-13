#!/usr/bin/env python3
"""Lint check ensuring all coding bots are registered correctly.

The script scans all Python modules in the repository and inspects classes whose
names end with ``Bot``.  Each bot class must be
decorated with ``@self_coding_managed``.  Modules that omit the decorator
are only allowed when they explicitly register the bot via
``BotRegistry.register_bot`` *and* log evaluations with ``db.log_eval``.
This accommodates factory-based or dynamically constructed bots.  Test
modules are ignored.
"""

from __future__ import annotations

import ast
from pathlib import Path


# Base classes that identify coding bots.
#
# Any class inheriting from one of these bases is treated as a bot even if its
# name does not end with ``Bot``.  The list can be extended as new bot base
# classes are introduced.
KNOWN_BOT_BASES = {"AdminBotBase"}


# Files where ``Bot`` classes are known to be configuration or helper objects
# rather than true coding bots.  These modules are skipped to avoid false
# positives.
EXCLUDED_PATHS = {
    Path("config.py"),
    Path("investment_engine.py"),
    Path("revenue_amplifier.py"),
    Path("plugins/metrics_prediction.py"),
    # ``data_bot.py`` is a metrics helper rather than a self-coding bot.
    Path("data_bot.py"),
}


def _inherits_bot_base(cls: ast.ClassDef) -> bool:
    """Return ``True`` if *cls* inherits from a known bot base."""

    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id in KNOWN_BOT_BASES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in KNOWN_BOT_BASES:
            return True
    return False


def _class_missing(cls: ast.ClassDef) -> bool:
    """Return ``True`` if *cls* lacks the ``self_coding_managed`` decorator."""
    for dec in cls.decorator_list:
        if isinstance(dec, ast.Call):
            dec = dec.func
        if isinstance(dec, ast.Name) and dec.id == "self_coding_managed":
            return False
        if isinstance(dec, ast.Attribute) and dec.attr == "self_coding_managed":
            return False
    return True


def _has_register_and_log(tree: ast.AST) -> bool:
    """Return ``True`` if module registers the bot and logs evaluations."""

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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders: list[tuple[Path, list[str]]] = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        rel = path.relative_to(root)
        if rel in EXCLUDED_PATHS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        missing = [
            node.name
            for node in getattr(tree, "body", [])
            if isinstance(node, ast.ClassDef)
            and (node.name.endswith("Bot") or _inherits_bot_base(node))
            and _class_missing(node)
        ]
        if missing and not _has_register_and_log(tree):
            offenders.append((rel, missing))
    if offenders:
        for path, classes in offenders:
            cls_list = ", ".join(classes)
            print(f"{path}: missing @self_coding_managed on {cls_list}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
