#!/usr/bin/env python3
"""Detect raw Stripe key or endpoint usage in tracked files."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamic_path_router import resolve_path  # noqa: E402
from stripe_detection import PAYMENT_KEYWORDS  # noqa: E402

REPO_ROOT = resolve_path(".")
# Exclude specific paths (resolved absolute)
EXCLUDED = {
    resolve_path("stripe_billing_router.py").resolve(),
    resolve_path("scripts/check_stripe_imports.py").resolve(),
    resolve_path("scripts/check_raw_stripe_usage.py").resolve(),
}
# Files under any of these directory names are ignored
EXCLUDED_DIRS = {"tests", "unit_tests", "fixtures", "finance_logs"}

PATTERN = re.compile(r"api\.stripe\.com|['\"](?:sk_|pk_)[^'\"]*['\"]")
KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in PAYMENT_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        p = (REPO_ROOT / line).resolve()
        if p in EXCLUDED:
            continue
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        try:
            with p.open("rb") as fh:
                sample = fh.read(1024)
                if b"\0" in sample:
                    continue
        except OSError:
            continue
        paths.append(p)
    return paths


def main() -> int:
    raw_offenders: list[str] = []
    keyword_offenders: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        kw_lines: list[tuple[int, str]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                raw_offenders.append(f"{rel}:{lineno}:{line.strip()}")
            if KEYWORD_PATTERN.search(line):
                kw_lines.append((lineno, line.strip()))
        if kw_lines and "stripe_billing_router" not in text:
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            for lineno, line in kw_lines:
                keyword_offenders.append(f"{rel}:{lineno}:{line}")
    if raw_offenders:
        print("Raw Stripe keys or endpoints detected:")
        for off in raw_offenders:
            print(off)
    if keyword_offenders:
        print("Payment keywords without stripe_billing_router detected:")
        for off in keyword_offenders:
            print(off)
    if raw_offenders or keyword_offenders:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
