"""Install dependencies required for the self-improvement subsystem.

This script installs the packages expected by :mod:`self_improvement.init`.
Run ``make install-self-improvement-deps`` to execute it.
"""
from __future__ import annotations

import subprocess
import sys

REQUIRED = [
    "quick_fix_engine>=1.0",
    "sandbox_runner>=1.0",
    "neurosales",
    "relevancy_radar",
    "torch>=2.0",
]


def main() -> None:
    cmd = [sys.executable, "-m", "pip", "install", *REQUIRED]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
