"""Scan the repository for bot classes missing @self_coding_managed.

The script walks all modules matching ``*_bot.py`` (excluding tests) and
reports any classes that end with ``Bot`` or inherit from known bot bases but
lack the ``@self_coding_managed`` decorator.  Modules that explicitly call
``BotRegistry.register_bot`` and log evaluations via ``log_eval`` are treated as
managed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

KNOWN_BOT_BASES = {"AdminBotBase"}

# Some modules share the ``Bot`` suffix but are infrastructure helpers rather
# than self-coding bots.  Listing them here prevents false positives when this
# script runs under pre-commit or the test suite.
EXCLUDED_PATHS = {
    Path("data_bot.py"),
    Path("prediction_manager_bot.py"),
    Path("bot_creation_bot.py"),
    Path("bot_development_bot.py"),
}


def _inherits_bot_base(cls: ast.ClassDef) -> bool:
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id in KNOWN_BOT_BASES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in KNOWN_BOT_BASES:
            return True
    return False


def _has_decorator(cls: ast.ClassDef) -> bool:
    return any(
        (isinstance(dec, ast.Name) and dec.id == "self_coding_managed")
        or (isinstance(dec, ast.Attribute) and dec.attr == "self_coding_managed")
        or (
            isinstance(dec, ast.Call)
            and (
                (isinstance(dec.func, ast.Name) and dec.func.id == "self_coding_managed")
                or (
                    isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "self_coding_managed"
                )
            )
        )
        for dec in cls.decorator_list
    )


def _has_register_and_log(tree: ast.AST) -> bool:
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


def _register_missing_refs(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "register_bot":
                kw = {k.arg for k in node.keywords if k.arg}
                if not {"manager", "data_bot"} <= kw:
                    return True
    return False


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[tuple[Path, list[str]]] = []
    bad_calls: list[Path] = []
    for path in root.rglob("*_bot.py"):
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
            and not _has_decorator(node)
        ]
        if missing and not _has_register_and_log(tree):
            offenders.append((path.relative_to(root), missing))
        if _register_missing_refs(tree):
            bad_calls.append(path.relative_to(root))
    if offenders or bad_calls:
        for path in bad_calls:
            print(f"{path}: register_bot missing manager/data_bot")
        for path, classes in offenders:
            print(f"{path}: unmanaged bot classes: {', '.join(classes)}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
