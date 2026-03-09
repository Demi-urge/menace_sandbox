from __future__ import annotations
"""Provision external dependencies for Menace."""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .local_infrastructure_provisioner import LocalInfrastructureProvisioner


@dataclass(frozen=True)
class DependencyProvisioningResult:
    """Outcome returned by :meth:`ExternalDependencyProvisioner.provision`."""

    status: str
    message: str = ""
    command: tuple[str, ...] = ()

    @property
    def is_success(self) -> bool:
        return self.status in {"provisioned", "managed_externally", "skipped"}

    @property
    def is_unavailable(self) -> bool:
        return self.status == "unavailable"


class ExternalDependencyProvisioner:
    """Start external services using Docker Compose if available."""

    def __init__(self, compose_file: str = "docker-compose.yml") -> None:
        self.compose_file = compose_file
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def provisioning_enabled(self) -> bool:
        """Return whether local provisioning should run on this host."""
        return not self._env_flag("MENACE_SKIP_EXTERNAL_DEP_PROVISIONING")

    def provisioning_optional(self) -> bool:
        """Return whether provisioning failures should degrade instead of crash."""
        return self._env_flag("MENACE_EXTERNAL_DEP_PROVISION_OPTIONAL")

    def _detect_compose_command(self) -> DependencyProvisioningResult:
        """Detect supported docker compose command flavor."""
        candidates = (("docker", "compose"), ("docker-compose",))
        probe_failures: list[str] = []
        for candidate in candidates:
            probe_cmd = [*candidate, "version"]
            proc = subprocess.run(probe_cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return DependencyProvisioningResult(
                    status="detected",
                    command=candidate,
                    message=(proc.stdout or proc.stderr).strip(),
                )
            details = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            probe_failures.append(f"{' '.join(probe_cmd)} -> {details}")

        detected_versions = []
        for version_cmd in (
            ["docker", "--version"],
            ["docker", "compose", "--version"],
            ["docker-compose", "--version"],
        ):
            proc = subprocess.run(version_cmd, capture_output=True, text=True)
            output = (proc.stdout or proc.stderr).strip() or "not available"
            detected_versions.append(f"{' '.join(version_cmd)} => {output}")
        return DependencyProvisioningResult(
            status="unavailable",
            message=(
                "No supported Docker Compose command found. "
                f"Attempted probes: {', '.join(probe_failures)}. "
                f"Detected versions: {'; '.join(detected_versions)}. "
                "Remediation: install Docker Compose v2 (`docker compose`) or legacy "
                "`docker-compose`, or set MENACE_EXTERNAL_DEPS_MANAGED_EXTERNALLY=1 "
                "to skip local provisioning."
            ),
        )

    def provision(self) -> DependencyProvisioningResult:
        """Ensure containers are running for required services."""
        if self._env_flag("MENACE_EXTERNAL_DEPS_MANAGED_EXTERNALLY"):
            return DependencyProvisioningResult(
                status="managed_externally",
                message="external dependencies managed externally",
            )

        if not self.provisioning_enabled():
            self.logger.info(
                "external dependency provisioning skipped by MENACE_SKIP_EXTERNAL_DEP_PROVISIONING"
            )
            return DependencyProvisioningResult(
                status="skipped",
                message="provisioning skipped",
            )

        prov = LocalInfrastructureProvisioner(self.compose_file)
        cfg = Path(prov.ensure_compose_file())
        detection = self._detect_compose_command()
        if detection.is_unavailable:
            return detection

        compose_cmd = list(detection.command)
        up_cmd = [*compose_cmd, "-f", str(cfg), "up", "-d"]
        down_cmd = [*compose_cmd, "-f", str(cfg), "down"]
        try:
            subprocess.run(
                up_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort
            self.logger.error(
                "dependency provisioning failed: command=%s\nerror: %s\nstdout: %s\nstderr: %s",
                " ".join(up_cmd),
                exc,
                exc.stdout,
                exc.stderr,
            )
            subprocess.run(
                down_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raise RuntimeError(
                "dependency provisioning failed. "
                f"Attempted command: {' '.join(up_cmd)}. "
                "Set MENACE_EXTERNAL_DEPS_MANAGED_EXTERNALLY=1 if dependencies "
                "are managed outside this host."
            ) from exc
        return DependencyProvisioningResult(
            status="provisioned",
            command=tuple(up_cmd),
            message="dependencies provisioned",
        )


__all__ = ["DependencyProvisioningResult", "ExternalDependencyProvisioner"]
