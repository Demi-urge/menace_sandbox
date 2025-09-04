from __future__ import annotations

"""Provision external services and monitor their health."""

import logging
import os
import subprocess
import time
import uuid

from db_router import init_db_router
from dynamic_path_router import resolve_path

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

from menace import RAISE_ERRORS  # noqa: E402

logger = logging.getLogger(__name__)

from menace.external_dependency_provisioner import (  # noqa: E402
    ExternalDependencyProvisioner,
)
from menace.dependency_watchdog import DependencyWatchdog  # noqa: E402


def _cleanup(compose_file: str) -> None:
    """Stop running containers and ignore errors."""
    try:
        subprocess.check_call(
            ["docker", "compose", "-f", compose_file, "down"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("cleaned up containers")
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("cleanup failed: %s", exc)


def _parse_map(value: str) -> dict[str, str]:
    pairs = [p.strip() for p in value.split(',') if p.strip()]
    result: dict[str, str] = {}
    for pair in pairs:
        if '=' in pair:
            k, v = pair.split('=', 1)
            result[k.strip()] = v.strip()
    return result


def main() -> None:
    """Provision dependencies and start the watchdog loop."""

    logging.basicConfig(level=logging.INFO)
    provisioner = ExternalDependencyProvisioner()
    try:
        provisioner.provision()

        endpoints = _parse_map(os.getenv("DEPENDENCY_ENDPOINTS", ""))
        backups = _parse_map(os.getenv("DEPENDENCY_BACKUPS", ""))
        watchdog = DependencyWatchdog(endpoints, backups)
        interval = float(os.getenv("WATCHDOG_INTERVAL", "60"))

        while True:
            watchdog.check()
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("interrupted, performing cleanup")
        _cleanup(provisioner.compose_file)
        if RAISE_ERRORS:
            raise


if __name__ == '__main__':
    main()
