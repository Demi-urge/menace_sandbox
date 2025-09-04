#!/usr/bin/env python3
"""Prevent direct ``sqlite3.connect`` usage outside approved modules.

The allow list is shared with ``tests/test_db_router_enforcement.py`` via
``tests/approved_sqlite3_usage.txt`` so CI and pre-commit use the same source
of truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dynamic_path_router import get_project_root, resolve_path

PATTERN = "sqlite3.connect("

# Resolve repository root so the check works regardless of the current working
# directory. Paths in ``ALLOWLIST`` are defined relative to this directory.
REPO_ROOT = get_project_root()


def _load_allowlist(root: Path) -> set[Path]:
    try:
        text = resolve_path("tests/approved_sqlite3_usage.txt").read_text(
            encoding="utf-8"
        )
    except OSError:
        return set()
    items: set[Path] = set()
    for line in text.splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            items.add((root / entry).resolve())
    return items


ALLOWLIST = _load_allowlist(REPO_ROOT)


def main() -> int:
    root = REPO_ROOT
    offenders: list[str] = []
    for filename in sys.argv[1:]:
        path = Path(filename)
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if path.suffix != ".py" or path in ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN in line:
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{lineno}:{line.strip()}")
    if offenders:
        print("Direct sqlite3.connect calls detected (use DBRouter):")
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
