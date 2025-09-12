#!/usr/bin/env python3
"""CLI wrapper around tools.check_self_coding_registration."""
from __future__ import annotations

import sys
from pathlib import Path

# ensure repository root on path when executed directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.check_self_coding_registration import main as _main


def main() -> int:  # pragma: no cover - thin wrapper
    return _main()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
