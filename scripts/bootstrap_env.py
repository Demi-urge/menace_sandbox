#!/usr/bin/env python3
"""Bootstrap the Menace environment and verify dependencies."""
from __future__ import annotations

import logging

from menace.startup_checks import run_startup_checks
from menace.environment_bootstrap import EnvironmentBootstrapper


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_startup_checks()
    EnvironmentBootstrapper().bootstrap()


if __name__ == "__main__":
    main()
