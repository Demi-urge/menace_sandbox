from __future__ import annotations

"""Lightweight environment bootstrap utilities used when MENACE_LIGHT_IMPORTS is
set."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - optional package form
    from .environment_bootstrap import EnvironmentBootstrapper as _FullBootstrapper
except Exception:  # pragma: no cover - fallback to local module
    from .environment_bootstrap import EnvironmentBootstrapper as _FullBootstrapper

from .retry_utils import retry

class EnvironmentBootstrapper:
    """Minimal bootstrapper providing remote deployment via SSH with basic
    environment preparation helpers."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.required_commands = [
            "git",
            "curl",
            "python3",
            "pip",
            "ssh",
            "alembic",
        ]
        self.required_os_packages = os.getenv("MENACE_OS_PACKAGES", "").split(",")
        self.secret_names = os.getenv("BOOTSTRAP_SECRET_NAMES", "").split(",")
        try:  # pragma: no cover - optional
            from .secrets_manager import SecretsManager
        except Exception:  # pragma: no cover - optional
            SecretsManager = None  # type: ignore
        try:  # pragma: no cover - optional
            from .vault_secret_provider import VaultSecretProvider
        except Exception:  # pragma: no cover - optional
            VaultSecretProvider = None  # type: ignore
        self.secrets = SecretsManager() if SecretsManager else None
        self.vault = (
            VaultSecretProvider() if VaultSecretProvider and os.getenv("SECRET_VAULT_URL") else None
        )

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        @retry(Exception, attempts=3)
        def _execute() -> subprocess.CompletedProcess:
            return subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        try:
            return _execute()
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.error("command failed after retries: %s", exc)
            raise

    # ------------------------------------------------------------------
    def check_commands(self, cmds: Iterable[str]) -> None:
        for cmd in cmds:
            if shutil.which(cmd) is None:
                self.logger.warning("required command missing: %s", cmd)

    # ------------------------------------------------------------------
    def check_os_packages(self, packages: Iterable[str]) -> None:
        if not packages:
            return
        cmd: list[str] | None = None
        if shutil.which("dpkg"):
            cmd = ["dpkg", "-s"]
        elif shutil.which("rpm"):
            cmd = ["rpm", "-q"]
        elif shutil.which("pacman"):
            cmd = ["pacman", "-Qi"]
        elif shutil.which("apk"):
            cmd = ["apk", "info", "-e"]
        else:
            self.logger.warning("no package manager found to verify packages")
            return
        missing: list[str] = []
        for pkg in packages:
            p = pkg.strip()
            if not p:
                continue
            try:
                subprocess.run(
                    cmd + [p],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except Exception:
                self.logger.error("required package missing: %s", p)
                missing.append(p)
        if missing:
            raise RuntimeError("missing OS packages: " + ", ".join(missing))

    # ------------------------------------------------------------------
    def export_secrets(self) -> None:
        return _FullBootstrapper.export_secrets(self)

    # ------------------------------------------------------------------
    def run_migrations(self) -> None:
        return _FullBootstrapper.run_migrations(self)

    def deploy_across_hosts(self, hosts: Iterable[str]) -> None:
        """Bootstrap remote hosts using SSH."""
        self.export_secrets()
        self.check_commands(self.required_commands)
        pkgs = [p.strip() for p in self.required_os_packages if p.strip()]
        if pkgs:
            try:
                self.check_os_packages(pkgs)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.error("OS package check failed: %s", exc)
        self.run_migrations()
        for host in hosts:
            h = str(host).strip()
            if not h:
                continue
            self.logger.info("bootstrapping %s", h)
            try:
                self._run(["ssh", h, "python3", "-m", "menace.environment_bootstrap"])
                self.logger.info("bootstrap succeeded on %s", h)
            except Exception:
                self.logger.error("bootstrap failed on %s", h)

__all__ = ["EnvironmentBootstrapper"]
