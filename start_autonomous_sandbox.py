"""Entry point for launching the autonomous sandbox.

This small wrapper adds a bit of resiliency around the sandbox bootstrap by
capturing startup exceptions and allowing the log level to be configured via
the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sandbox_runner.bootstrap import launch_sandbox


def main(argv: list[str] | None = None) -> None:
    """Launch the sandbox with optional log level configuration.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. If ``None`` the arguments will
        be pulled from :data:`sys.argv`.
    """

    parser = argparse.ArgumentParser(description="Launch the autonomous sandbox")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    args = parser.parse_args(argv)

    # Configure logging; errors are always shown but the level can be raised
    # for debugging during initialization.
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), None))

    try:
        launch_sandbox()
    except Exception:  # pragma: no cover - defensive catch
        logging.exception("Failed to launch sandbox")
        sys.exit(1)


if __name__ == "__main__":
    main()
