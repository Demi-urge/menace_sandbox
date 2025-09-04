#!/usr/bin/env python3
"""Detect direct Stripe SDK imports outside ``stripe_billing_router.py``."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {(REPO_ROOT / "stripe_billing_router.py").resolve()}  # path-ignore
PATTERN = re.compile(r"^\s*(?:import stripe|from stripe\b)")


def main() -> int:
    offenders: list[str] = []
    for filename in sys.argv[1:]:
        path = Path(filename)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        else:
            path = path.resolve()
        if path.suffix != ".py" or path in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{lineno}:{line.strip()}")
    if offenders:
        print("Direct Stripe imports detected (use stripe_billing_router):")
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
