"""Fail if a coding bot lacks ``@self_coding_managed`` or ``internalize_coding_bot``.

The script walks all Python sources looking for modules that define classes
resembling coding bots.  Any module that imports ``SelfCodingEngine`` or calls
``internalize_coding_bot`` must both decorate its bot classes with
``@self_coding_managed`` and invoke ``internalize_coding_bot`` to register the
bot.  Test files are ignored.
"""

from __future__ import annotations

import ast
from pathlib import Path


# Base classes that identify coding bots even if their name does not end with
# ``Bot``.  Extend this set as new base classes are introduced.
KNOWN_BOT_BASES = {"AdminBotBase"}

# Files where classes using "Bot" in the name are helpers rather than real
# coding bots.  These modules are skipped to avoid false positives.
EXCLUDED_PATHS = {
    Path("config.py"),
    Path("investment_engine.py"),
    Path("revenue_amplifier.py"),
    Path("plugins/metrics_prediction.py"),
    # ``data_bot.py`` is a metrics helper rather than a self-coding bot.
    Path("data_bot.py"),
}


def _imports_self_coding_engine(tree: ast.AST) -> bool:
    """Return ``True`` if the module imports ``SelfCodingEngine``."""

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "SelfCodingEngine" for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name == "SelfCodingEngine" for alias in node.names):
                return True
    return False


def _inherits_bot_base(cls: ast.ClassDef) -> bool:
    """Return ``True`` if *cls* inherits from a known bot base."""

    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id in KNOWN_BOT_BASES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in KNOWN_BOT_BASES:
            return True
    return False


def _has_decorator(cls: ast.ClassDef) -> bool:
    """Return ``True`` if *cls* has the ``self_coding_managed`` decorator."""

    for dec in cls.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "self_coding_managed":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "self_coding_managed":
            return True
    return False


def _has_internalize_call(tree: ast.AST) -> bool:
    """Return ``True`` if the module calls ``internalize_coding_bot``."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "internalize_coding_bot":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "internalize_coding_bot":
                return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders: list[tuple[Path, list[str], bool]] = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or "unit_tests" in path.parts:
            continue
        rel = path.relative_to(root)
        if rel in EXCLUDED_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            continue
        classes = [
            node
            for node in getattr(tree, "body", [])
            if isinstance(node, ast.ClassDef)
            and (node.name.endswith("Bot") or _inherits_bot_base(node))
        ]
        if not classes:
            continue
        if not (_imports_self_coding_engine(tree) or _has_internalize_call(tree)):
            continue
        missing = [node.name for node in classes if not _has_decorator(node)]
        internalized = _has_internalize_call(tree)
        if missing or not internalized:
            offenders.append((rel, missing, internalized))
    if offenders:
        for path, classes, internalized in offenders:
            if classes:
                cls_list = ", ".join(classes)
                print(f"{path}: missing @self_coding_managed on {cls_list}")
            if not internalized:
                print(f"{path}: missing internalize_coding_bot")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
