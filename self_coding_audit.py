#!/usr/bin/env python3
"""Audit bot classes and patch provenance.

Walks the repository to find classes whose names end with ``Bot`` lacking the
``@self_coding_managed`` decorator.  With ``--verify`` the script also checks
recent git commits for associated patch provenance recorded via
``SelfCodingManager.register_patch_cycle``.

Usage:
    python self_coding_audit.py
    python self_coding_audit.py --verify
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple
import argparse
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from menace_sandbox.patch_provenance import PatchProvenanceService

DECORATOR = "self_coding_managed"
EXCLUDED_PATHS = {
    Path("config.py"),
    Path("investment_engine.py"),
    Path("revenue_amplifier.py"),
    Path("plugins/metrics_prediction.py"),
    Path("data_bot.py"),
    Path("database_manager.py"),
}


def _has_decorator(node: ast.ClassDef) -> bool:
    """Return True if ``node`` uses ``@self_coding_managed``."""
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Name) and target.id == DECORATOR:
            return True
        if isinstance(target, ast.Attribute) and target.attr == DECORATOR:
            return True
    return False


def _has_register_and_log(tree: ast.AST) -> bool:
    """Return True if module registers the bot and logs evaluations."""

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


def _is_relevant_class(node: ast.ClassDef) -> bool:
    """Return True if class name ends with ``Bot``."""
    return node.name.endswith("Bot")


def _iter_python_files(root: Path) -> Iterable[Path]:
    """Yield all ``.py`` files under ``root`` except this script."""
    for path in root.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if "tests" in parts or "unit_tests" in parts:
            continue
        if rel in EXCLUDED_PATHS:
            continue
        yield path


def find_unmanaged_bots(root: Path) -> List[Tuple[Path, str, int]]:
    """Return list of unmanaged bot classes as ``(path, name, lineno)``."""
    unmanaged: List[Tuple[Path, str, int]] = []
    for path in _iter_python_files(root):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except Exception:
            # Skip files we cannot parse
            continue
        has_reg_log = _has_register_and_log(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_relevant_class(node):
                if not _has_decorator(node) and not has_reg_log:
                    unmanaged.append((path, node.name, node.lineno))
    return unmanaged


def _verify_commits(limit: int) -> int:
    """Verify recent commits have patch provenance."""

    os.environ.setdefault("PATCH_HISTORY_DB_PATH", str(ROOT / "patch_history.db"))
    svc = PatchProvenanceService()
    revs = subprocess.check_output(
        ["git", "rev-list", f"--max-count={limit}", "HEAD"], text=True
    ).splitlines()
    missing: List[str] = []
    for commit in revs:
        meta = svc.get(commit)
        if not meta or "patch_id" not in meta:
            missing.append(commit)
    if missing:
        print("Commits missing patch provenance:")
        for c in missing:
            print(f" - {c}")
        return 1
    print("All recent commits have patch provenance.")
    return 0


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit self-coding compliance")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify recent commits have registered patch cycles",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="number of recent commits to scan",
    )
    args = parser.parse_args(argv)

    if args.verify:
        sys.exit(_verify_commits(args.limit))

    root = Path(__file__).resolve().parent
    unmanaged = find_unmanaged_bots(root)
    if unmanaged:
        print("Found unmanaged bot classes:")
        for path, name, lineno in unmanaged:
            rel_path = path.relative_to(root)
            print(f" - {rel_path}:{lineno} -> {name} missing @self_coding_managed")
        print("Consider applying @self_coding_managed to the classes above.")
        sys.exit(1)
    print("All bot classes are managed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
