#!/usr/bin/env python3
"""Detect direct Stripe SDK imports or live keys in the repository."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {(REPO_ROOT / "stripe_billing_router.py").resolve()}  # path-ignore
IMPORT_PATTERN = re.compile(r"^\s*(?:import stripe|from stripe\b)")
KEY_PATTERN = re.compile(
    r"sk_live_[0-9A-Za-z]{8,}|pk_live_[0-9A-Za-z]{8,}|https://api\.stripe\.com"
)


def _check_imports(paths: list[Path]) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        if path.suffix != ".py" or path in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if IMPORT_PATTERN.search(line):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{lineno}:{line.strip()}")
    if offenders:
        print("Direct Stripe imports detected (use stripe_billing_router):")
    return offenders


def _check_keys(paths: list[Path]) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if KEY_PATTERN.search(line):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{lineno}:{line.strip()}")
    if offenders:
        print("Stripe live keys or endpoints detected:")
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", action="store_true", help="scan for Stripe keys")
    args, files = parser.parse_known_args(argv)

    paths: list[Path] = []
    for filename in files:
        p = Path(filename)
        paths.append(p if p.is_absolute() else (REPO_ROOT / p).resolve())

    offenders = _check_keys(paths) if args.keys else _check_imports(paths)
    if offenders:
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
