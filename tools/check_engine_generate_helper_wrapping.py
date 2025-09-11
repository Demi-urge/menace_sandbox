#!/usr/bin/env python3
"""Pre-commit check forbidding direct engine.generate_helper usage.

This script scans Python files for references to ``engine.generate_helper``
which should instead be invoked through ``manager_generate_helper``.  Test
modules and a small set of approved files are ignored.  Any occurrence of the
pattern elsewhere triggers a failure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ALLOWED_FILES = {
    "coding_bot_interface.py",  # defines manager_generate_helper
    "quick_fix_engine.py",      # legacy usage with explicit fallback
}
SCRIPT_NAME = Path(__file__).name


def check_file(path: str) -> bool:
    p = Path(path)
    if p.name in ALLOWED_FILES or p.name == SCRIPT_NAME:
        return True
    if any(part in {"tests", "unit_tests"} for part in p.parts):
        return True
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return True
    if "engine.generate_helper" in text:
        print(f"{p}: direct engine.generate_helper usage is forbidden")
        return False
    return True


def main(argv: Iterable[str]) -> int:
    ok = True
    for file_path in argv:
        if not check_file(file_path):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main(sys.argv[1:]))
