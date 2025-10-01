#!/usr/bin/env python3
"""Bootstrap the Menace environment and verify dependencies."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from menace.startup_checks import run_startup_checks
from menace.environment_bootstrap import EnvironmentBootstrapper


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Explicitly disable safe mode regardless of existing variables
    os.environ["MENACE_SAFE"] = "0"
    run_startup_checks()
    EnvironmentBootstrapper().bootstrap()


if __name__ == "__main__":
    main()
