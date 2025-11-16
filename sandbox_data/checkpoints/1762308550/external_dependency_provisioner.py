from __future__ import annotations
"""Provision external dependencies for Menace."""

import logging
import subprocess
from pathlib import Path

from .local_infrastructure_provisioner import LocalInfrastructureProvisioner


class ExternalDependencyProvisioner:
    """Start external services using Docker Compose if available."""

    def __init__(self, compose_file: str = "docker-compose.yml") -> None:
        self.compose_file = compose_file
        self.logger = logging.getLogger(self.__class__.__name__)

    def provision(self) -> None:
        """Ensure containers are running for required services."""
        prov = LocalInfrastructureProvisioner(self.compose_file)
        cfg = Path(prov.ensure_compose_file())
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(cfg), "up", "-d"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort
            self.logger.error(
                "dependency provisioning failed: %s\nstdout: %s\nstderr: %s",
                exc,
                exc.stdout,
                exc.stderr,
            )
            subprocess.run(
                ["docker", "compose", "-f", str(cfg), "down"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raise RuntimeError("dependency provisioning failed") from exc


__all__ = ["ExternalDependencyProvisioner"]

