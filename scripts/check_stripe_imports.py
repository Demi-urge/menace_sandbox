#!/usr/bin/env python3
"""Detect direct Stripe SDK imports or live keys in the repository."""
from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {
    (REPO_ROOT / "stripe_billing_router.py").resolve(),
    (REPO_ROOT / "scripts/check_stripe_imports.py").resolve(),
    (REPO_ROOT / "startup_checks.py").resolve(),
    (REPO_ROOT / "bot_development_bot.py").resolve(),
    (REPO_ROOT / "config_loader.py").resolve(),
    (REPO_ROOT / "codex_output_analyzer.py").resolve(),
}  # path-ignore
IMPORT_PATTERN = re.compile(r"^\s*(?:import stripe|from stripe\b)")
ROUTER_IMPORT = re.compile(
    r"^\s*(?:from\s+[\.\w]+\s+import\s+stripe_billing_router\b|import\s+stripe_billing_router\b)",
    re.MULTILINE,
)
KEYWORDS = {
    "stripe",
    "checkout",
    "billing",
    "invoice",
    "subscription",
    "payout",
    "charge",
}
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


def _check_keywords(paths: list[Path]) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        if path.suffix != ".py" or path in ALLOWED:
            continue
        if any(part in {"tests", "unit_tests", "fixtures", "finance_logs"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if ROUTER_IMPORT.search(text):
            continue
        try:
            tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        except tokenize.TokenError:
            continue
        for tok in tokens:
            if tok.type != tokenize.NAME:
                continue
            name = tok.string
            if name.isupper() or name == "stripe_billing_router":
                continue
            parts = name.lower().split("_")
            if any(part in KEYWORDS for part in parts):
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                offenders.append(f"{rel}:{tok.start[0]}:missing stripe_billing_router import")
                break
    if offenders:
        print(
            "Payment/Stripe keywords without stripe_billing_router import detected:"
        )
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

    if args.keys:
        offenders = _check_keys(paths)
    else:
        offenders = _check_imports(paths)
        offenders.extend(_check_keywords(paths))
    if offenders:
        for off in offenders:
            print(off)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
