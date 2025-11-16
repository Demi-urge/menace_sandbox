#!/usr/bin/env python3
"""Pre-commit check forbidding direct SelfCodingEngine patch usage.

The hook scans Python files for references to ``SelfCodingEngine.apply_patch``
or calls to ``apply_patch_with_retry``.  These are only permitted inside
``self_coding_manager.py`` and ``quick_fix_engine.py``.  Test modules are
ignored.  Any occurrence elsewhere triggers a failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ALLOWED_FILES = {"self_coding_manager.py", "quick_fix_engine.py"}
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
    if "SelfCodingEngine.apply_patch" in text or ".apply_patch_with_retry" in text:
        print(f"{p}: direct SelfCodingEngine patch usage is forbidden")
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
