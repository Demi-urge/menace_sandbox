"""Entry point for launching the autonomous sandbox.

This small wrapper adds a bit of resiliency around the sandbox bootstrap by
capturing startup exceptions and allowing the log level to be configured via
``SandboxSettings`` or overridden on the command line.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

from logging_utils import get_logger, setup_logging, set_correlation_id, log_record
from sandbox_settings import SandboxSettings
from sandbox_runner.bootstrap import (
    bootstrap_environment,
    launch_sandbox,
    sandbox_health,
    shutdown_autonomous_sandbox,
)
try:  # pragma: no cover - allow package relative import
    from metrics_exporter import (
        sandbox_restart_total,
        sandbox_crashes_total,
        sandbox_last_failure_ts,
    )
except Exception:  # pragma: no cover - fallback when run as a module
    from .metrics_exporter import (  # type: ignore
        sandbox_restart_total,
        sandbox_crashes_total,
        sandbox_last_failure_ts,
    )


def main(argv: list[str] | None = None) -> None:
    """Launch the sandbox with optional log level configuration.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. If ``None`` the arguments will
        be pulled from :data:`sys.argv`.
    """

    settings = SandboxSettings()

    parser = argparse.ArgumentParser(description="Launch the autonomous sandbox")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=settings.sandbox_log_level,
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run sandbox health checks and exit",
    )
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    cid = f"sas-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger = get_logger(__name__)
    sandbox_restart_total.labels(service="start_autonomous", reason="launch").inc()
    logger.info("sandbox start", extra=log_record(event="start"))

    try:
        if args.health_check:
            bootstrap_environment()
            logger.info(
                "sandbox health", extra=log_record(health=sandbox_health())
            )
            shutdown_autonomous_sandbox()
            logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
            return
        launch_sandbox()
        logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
    except Exception:  # pragma: no cover - defensive catch
        sandbox_crashes_total.inc()
        sandbox_last_failure_ts.set(time.time())
        logger.exception("sandbox failure", extra=log_record(event="failure"))
        sys.exit(1)
    finally:
        set_correlation_id(None)


if __name__ == "__main__":
    main()
