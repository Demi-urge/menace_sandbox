#!/usr/bin/env python3
"""Pre-commit check for unwrapped engine.generate_helper usage.

This script scans Python files for occurrences of ``engine.generate_helper(``
which should only appear inside :mod:`coding_bot_interface`'s
``manager_generate_helper`` wrapper.  Any other occurrence indicates direct
usage of the self-coding engine and triggers a failure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ALLOWED_FILES = {"coding_bot_interface.py"}
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
    if "engine.generate_helper(" in text:
        print(
            f"{p}: engine.generate_helper should be wrapped by manager_generate_helper"
        )
        return False
    return True


def main(argv: Iterable[str]) -> int:
    ok = True
    for file_path in argv:
        if not check_file(file_path):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
