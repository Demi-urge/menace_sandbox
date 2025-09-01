from __future__ import annotations

"""Bootstrap the sandbox and launch a self-improvement cycle."""

import logging

from sandbox_settings import SandboxSettings
from sandbox_runner.bootstrap import (
    bootstrap_environment,
    _verify_required_dependencies,
)
from self_improvement.api import init_self_improvement, start_self_improvement_cycle


logger = logging.getLogger(__name__)


def main() -> int:
    """Bootstrap dependencies and run the self-improvement cycle."""
    logging.basicConfig(level=logging.INFO)
    logger.info("validating sandbox environment and dependencies")
    settings = SandboxSettings()
    try:
        settings = bootstrap_environment(settings, _verify_required_dependencies)
    except SystemExit as exc:
        logger.error("dependency verification failed: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected environment failure
        logger.exception("environment bootstrap failed")
        return 1

    try:
        init_self_improvement(settings)
    except Exception as exc:  # pragma: no cover - missing helper packages
        logger.exception("self-improvement initialisation failed")
        return 1

    logger.info(
        "starting self-improvement cycle (repo=%s data_dir=%s)",
        settings.sandbox_repo_path,
        settings.sandbox_data_dir,
    )
    thread = start_self_improvement_cycle({"bootstrap": lambda: None})
    thread.start()
    logger.info("self-improvement cycle running; press Ctrl+C to stop")
    try:
        thread.join()
    except KeyboardInterrupt:
        logger.info("stopping self-improvement cycle")
        thread.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
