#!/usr/bin/env python3
"""Prevent direct sqlite3.connect usage outside db_router.py."""
from __future__ import annotations

import sys
from pathlib import Path

PATTERN = "sqlite3.connect("

# Resolve repository root so the check works regardless of the current working
# directory. Paths in ``ALLOWLIST`` are defined relative to this directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Files below are permitted to call sqlite3.connect directly. Any additional
# exceptions must be documented in ``docs/db_router.md`` and added here with
# explicit approval.
ALLOWLIST = {
    REPO_ROOT / "db_router.py",
    REPO_ROOT / "scripts/check_sqlite_connections.py",
    REPO_ROOT / "scripts/new_db.py",
    REPO_ROOT / "scripts/new_db_template.py",
    REPO_ROOT / "scripts/scaffold_db.py",
    REPO_ROOT / "scripts/new_vector_module.py",
    REPO_ROOT / "sync_shared_db.py",
}


def main() -> int:
    offenders: list[str] = []
    for filename in sys.argv[1:]:
        path = Path(filename)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
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
                    rel = path.relative_to(REPO_ROOT)
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
