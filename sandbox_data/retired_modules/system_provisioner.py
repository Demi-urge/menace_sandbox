from __future__ import annotations

"""Self-provisioning helper for missing system packages."""

import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


class SystemProvisioner:
    """Install packages and ensure container build scripts exist."""

    def __init__(self, *, packages: Iterable[str] | None = None) -> None:
        self.packages = list(packages or [])
        self.logger = logging.getLogger(self.__class__.__name__)
        self.manager = self._detect_package_manager()

    # ------------------------------------------------------------------
    def _detect_package_manager(self) -> str | None:
        """Determine the appropriate package manager for the system."""
        system = platform.system().lower()
        if system == "windows" and shutil.which("choco"):
            return "choco"
        if system == "darwin" and shutil.which("brew"):
            return "brew"

        # assume Linux variants
        if shutil.which("apt-get") and shutil.which("dpkg"):
            return "apt"
        if shutil.which("yum"):
            return "yum"
        if shutil.which("dnf"):
            return "dnf"
        if shutil.which("apk"):
            return "apk"
        return None

    # ------------------------------------------------------------------
    def ensure_packages(self) -> None:
        if not self.manager:
            self.logger.error("no supported package manager found")
            return
        for pkg in self.packages:
            if not self._installed(pkg):
                self._install(pkg)

    def _installed(self, pkg: str) -> bool:
        try:
            if self.manager == "apt":
                res = subprocess.run(["dpkg", "-s", pkg], capture_output=True)
                return res.returncode == 0
            if self.manager in {"yum", "dnf"}:
                res = subprocess.run(["rpm", "-q", pkg], capture_output=True)
                return res.returncode == 0
            if self.manager == "apk":
                res = subprocess.run(["apk", "info", pkg], capture_output=True)
                return res.returncode == 0
            if self.manager == "brew":
                res = subprocess.run(["brew", "list", pkg], capture_output=True)
                return res.returncode == 0
            if self.manager == "choco":
                res = subprocess.run(
                    ["choco", "list", "--local-only", pkg], capture_output=True
                )
                return res.returncode == 0 and pkg.lower() in res.stdout.decode().lower()
        except Exception as exc:
            self.logger.error("package check failed for %s: %s", pkg, exc)
        return False

    def _install(self, pkg: str) -> None:
        self.logger.info("installing %s via %s", pkg, self.manager)
        try:
            if self.manager == "apt":
                subprocess.run(["apt-get", "update"], check=False)
                subprocess.run(["apt-get", "install", "-y", pkg], check=True)
            elif self.manager == "yum":
                subprocess.run(["yum", "install", "-y", pkg], check=True)
            elif self.manager == "dnf":
                subprocess.run(["dnf", "install", "-y", pkg], check=True)
            elif self.manager == "apk":
                subprocess.run(["apk", "add", pkg], check=True)
            elif self.manager == "brew":
                subprocess.run(["brew", "install", pkg], check=True)
            elif self.manager == "choco":
                subprocess.run(["choco", "install", "-y", pkg], check=True)
            else:
                raise RuntimeError("unsupported package manager")
        except Exception as exc:
            self.logger.exception("failed to install %s: %s", pkg, exc)
            raise RuntimeError(f"installation of {pkg} failed") from exc

    # ------------------------------------------------------------------
    def ensure_dockerfile(self, path: str = "Dockerfile") -> None:
        p = Path(path)
        if not p.exists():
            p.write_text("FROM python:3.11-slim\nCOPY . /app\nWORKDIR /app\n")


__all__ = ["SystemProvisioner"]
