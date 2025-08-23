#!/usr/bin/env python3
"""Prevent direct sqlite3.connect usage outside db_router.py."""
from __future__ import annotations

import sys
from pathlib import Path

PATTERN = "sqlite3.connect("

# Files below are permitted to call sqlite3.connect directly. Any additional
# exceptions must be documented in ``docs/db_router.md`` and added here with
# explicit approval.
ALLOWLIST = {
    Path("db_router.py").resolve(),
    Path(__file__).resolve(),
    Path("scripts/new_db.py").resolve(),
    Path("scripts/new_db_template.py").resolve(),
    Path("scripts/scaffold_db.py").resolve(),
    Path("scripts/new_vector_module.py").resolve(),
}


def main() -> int:
    offenders: list[str] = []
    for filename in sys.argv[1:]:
        path = Path(filename).resolve()
        if path in ALLOWLIST or path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN in line:
                rel = path.relative_to(Path.cwd())
                offenders.append(f"{rel}:{lineno}:{line.strip()}")
    if offenders:
        print("Direct sqlite3.connect calls detected (use DBRouter):")
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

