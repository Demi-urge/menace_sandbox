#!/usr/bin/env python3
"""Verify essential environment variables for running the personal sandbox."""

from __future__ import annotations

import os
import sys

# Variables mentioned in docs/autonomous_sandbox.md
_VARS = [
    "AUTO_SANDBOX",
    "SANDBOX_CYCLES",
    "SANDBOX_ROI_TOLERANCE",
    "RUN_CYCLES",
    "AUTO_DASHBOARD_PORT",
    "EXPORT_SYNERGY_METRICS",
    "SYNERGY_METRICS_PORT",
    "SANDBOX_DATA_DIR",
    "PATCH_SCORE_BACKEND_URL",
    "DATABASE_URL",
    "BOT_DB_PATH",
    "BOT_PERFORMANCE_DB",
    "MAINTENANCE_DB",
    "OPENAI_API_KEY",
]

# Critical variables that must be set
_CRITICAL = {"OPENAI_API_KEY"}


def main(argv: list[str] | None = None) -> None:
    missing_critical: list[str] = []
    for name in _VARS:
        if not os.getenv(name):
            print(f"Warning: {name} is not set", file=sys.stderr)
            if name in _CRITICAL:
                missing_critical.append(name)
    if missing_critical:
        print(
            "Critical variables missing: " + ", ".join(missing_critical),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
