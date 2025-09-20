"""Automated environment bootstrap utilities.

The bootstrapper verifies system requirements, installs optional
dependencies and now exports secrets listed in ``BOOTSTRAP_SECRET_NAMES``
via :class:`SecretsManager`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import shutil
import importlib.util
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
import threading
import json

from .config_discovery import ensure_config, ConfigDiscovery

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .cluster_supervisor import ClusterServiceSupervisor
from .infrastructure_bootstrap import InfrastructureBootstrapper
from .retry_utils import retry
from .system_provisioner import SystemProvisioner
from .secrets_manager import SecretsManager
from .vault_secret_provider import VaultSecretProvider
from .external_dependency_provisioner import ExternalDependencyProvisioner
from . import startup_checks
from .vector_service.embedding_scheduler import start_scheduler_from_env

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore


class EnvironmentBootstrapper:
    """Bootstrap dependencies and infrastructure on startup."""

    def __init__(
        self,
        *,
        tf_dir: str | None = None,
        vault: VaultSecretProvider | None = None,
        cluster_supervisor: "ClusterServiceSupervisor" | None = None,
    ) -> None:
        disc = ConfigDiscovery()
        disc.discover()
        interval = os.getenv("CONFIG_DISCOVERY_INTERVAL")
        if interval:
            try:
                sec = float(interval)
            except ValueError:
                sec = 0.0
            if sec > 0:
                disc.run_continuous(interval=sec)
        self.tf_dir = tf_dir or os.getenv("TERRAFORM_DIR")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bootstrapper = InfrastructureBootstrapper(self.tf_dir)
        self._threads: list[threading.Thread] = []
        self.required_commands = ["git", "curl", "python3"]
        self.remote_endpoints = os.getenv("MENACE_REMOTE_ENDPOINTS", "").split(",")
        self.required_os_packages = os.getenv("MENACE_OS_PACKAGES", "").split(",")
        self.secrets = SecretsManager()
        self.vault = vault or (VaultSecretProvider() if os.getenv("SECRET_VAULT_URL") else None)
        self.secret_names = os.getenv("BOOTSTRAP_SECRET_NAMES", "").split(",")
        self.min_driver_version = os.getenv("MIN_NVIDIA_DRIVER_VERSION")
        self.strict_driver_check = os.getenv("STRICT_NVIDIA_DRIVER_CHECK") == "1"
        self.cluster_sup = cluster_supervisor
        if self.cluster_sup:
            hosts = [h.strip() for h in os.getenv("CLUSTER_HOSTS", "").split(",") if h.strip()]
            if hosts:
                try:
                    self.cluster_sup.add_hosts(hosts)
                    self.cluster_sup.start_all()
                except Exception as exc:  # pragma: no cover - remote may fail
                    self.logger.error("cluster supervisor failed: %s", exc)

    def _run(self, cmd: list[str], **kwargs) -> None:
        """Run a subprocess command with retries."""

        @retry(Exception, attempts=3)
        def _execute() -> subprocess.CompletedProcess:
            return subprocess.run(cmd, check=True, **kwargs)

        try:
            _execute()
        except Exception as exc:
            self.logger.error("command failed after retries: %s", exc)
            raise

    # ------------------------------------------------------------------
    def check_commands(self, cmds: Iterable[str]) -> None:
        """Verify that required commands are available."""
        for cmd in cmds:
            if shutil.which(cmd) is None:
                self.logger.warning("required command missing: %s", cmd)

    # ------------------------------------------------------------------
    def check_nvidia_driver(self) -> None:
        """Ensure the installed NVIDIA driver meets the minimum version."""
        if not self.min_driver_version:
            return
        if shutil.which("nvidia-smi") is None:
            self.logger.warning("nvidia-smi not found; skipping driver check")
            return
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            ver_str = out.splitlines()[0].strip()
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("failed to query nvidia driver: %s", exc)
            return

        try:
            from packaging import version as packaging_version

            current = packaging_version.parse(ver_str)
            minimum = packaging_version.parse(self.min_driver_version)
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("driver version parse failed: %s", exc)
            return

        if current < minimum:
            msg = f"NVIDIA driver {current} < required {minimum}"
            if self.strict_driver_check:
                raise RuntimeError(msg)
            self.logger.warning(msg)

    # ------------------------------------------------------------------
    def check_remote_dependencies(self, urls: Iterable[str]) -> None:
        """Ensure remote services are reachable."""
        missing = False
        for url in urls:
            u = url.strip()
            if not u:
                continue
            try:
                subprocess.run(
                    ["curl", "-I", "--max-time", "2", u],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except Exception as exc:
                self.logger.error(
                    "remote dependency unreachable: %s - %s", u, exc
                )
                missing = True
        if missing:
            ExternalDependencyProvisioner().provision()

    # ------------------------------------------------------------------
    def check_os_packages(self, packages: Iterable[str]) -> None:
        """Verify that required system packages are installed."""
        if not packages:
            return
        cmd: list[str] | None = None
        if shutil.which("dpkg"):
            cmd = ["dpkg", "-s"]
        elif shutil.which("rpm"):
            cmd = ["rpm", "-q"]
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
        """Expose configured secrets as environment variables."""
        for name in self.secret_names:
            n = name.strip()
            if not n:
                continue
            try:
                if self.vault:
                    self.vault.export_env(n)
                else:
                    self.secrets.export_env(n)
            except Exception as exc:  # pragma: no cover - log only
                self.logger.error("export secret %s failed: %s", n, exc)

    # ------------------------------------------------------------------
    def deploy_across_hosts(self, hosts: Iterable[str]) -> None:
        """Run the bootstrap script on remote hosts via SSH."""
        for host in hosts:
            h = host.strip()
            if not h:
                continue
            try:
                self.logger.info("remote bootstrap on %s", h)
                subprocess.run(
                    ["ssh", h, "python3", "-m", "menace.environment_bootstrap"],
                    check=True,
                )
            except Exception as exc:
                self.logger.error("remote bootstrap failed for %s: %s", h, exc)

    # ------------------------------------------------------------------
    def bootstrap_vector_assets(self) -> None:
        """Download model and seed default ranking weights."""
        try:
            from .vector_service import download_model as _dm

            dest = resolve_path(
                "vector_service/minilm/tiny-distilroberta-base.tar.xz"
            )
            if not dest.exists():
                _dm.bundle(dest)
        except Exception as exc:  # pragma: no cover - log only
            self.logger.warning("embedding model download failed: %s", exc)

        try:
            reg_path = resolve_path(
                "vector_service/embedding_registry.json"
            )
            with open(reg_path, "r", encoding="utf-8") as fh:
                names = list(json.load(fh).keys())
        except Exception:
            names = []

        if names:
            try:
                from .vector_metrics_db import VectorMetricsDB

                vdb = VectorMetricsDB("vector_metrics.db")
                if not vdb.get_db_weights():
                    vdb.set_db_weights({n: 1.0 for n in names})
                vdb.conn.close()
            except Exception as exc:  # pragma: no cover - log only
                self.logger.warning("VectorMetricsDB bootstrap failed: %s", exc)

            hist = resolve_path("sandbox_data") / "roi_history.json"
            if not hist.exists():
                try:
                    hist.parent.mkdir(parents=True, exist_ok=True)
                    with open(hist, "w", encoding="utf-8") as fh:
                        json.dump({"origin_db_deltas": {n: [0.0] for n in names}}, fh)
                except Exception as exc:  # pragma: no cover - log only
                    self.logger.warning("ROITracker bootstrap failed: %s", exc)

    # ------------------------------------------------------------------
    def install_dependencies(self, requirements: Iterable[str]) -> None:
        for req in requirements:
            try:
                self._run(["pip", "install", req])
            except Exception as exc:  # pragma: no cover - log only
                self.logger.error("failed installing %s: %s", req, exc)

    # ------------------------------------------------------------------
    def run_migrations(self) -> None:
        if not Path("alembic.ini").exists():
            return
        try:
            self._run(["alembic", "upgrade", "head"])
        except Exception as exc:  # pragma: no cover - log only
            self.logger.error("migrations failed: %s", exc)

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        ensure_config()
        self.export_secrets()
        self.check_commands(self.required_commands)
        self.check_nvidia_driver()
        pkgs = [p.strip() for p in self.required_os_packages if p.strip()]
        if pkgs:
            try:
                self.check_os_packages(pkgs)
            except RuntimeError as exc:
                msg = str(exc)
                if "missing OS packages:" in msg:
                    missing = [p.strip() for p in msg.split(":", 1)[1].split(",")]
                    SystemProvisioner(packages=missing).ensure_packages()
                    self.check_os_packages(pkgs)
                else:
                    raise
        self.check_remote_dependencies(self.remote_endpoints)
        missing = startup_checks.verify_project_dependencies()
        if missing:
            self.install_dependencies(missing)
        if importlib.util.find_spec("apscheduler") is None:
            self.install_dependencies(["apscheduler"])
        if shutil.which("systemctl"):
            try:
                subprocess.run(
                    ["systemctl", "enable", "--now", "sandbox_autopurge.timer"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except Exception as exc:  # pragma: no cover - log only
                self.logger.warning(
                    "failed enabling sandbox_autopurge.timer: %s", exc
                )
        else:
            self.logger.info(
                "systemctl not available; skipping sandbox_autopurge timer activation"
            )
        deps = os.getenv("MENACE_BOOTSTRAP_DEPS", "").split(",")
        deps = [d.strip() for d in deps if d.strip()]
        if deps:
            self.install_dependencies(deps)
        self.run_migrations()
        self.bootstrapper.bootstrap()
        # The security auditor previously enforced Bandit and Safety checks
        # and enabled safe mode when they failed. These checks have been
        # removed, so the bootstrapper no longer toggles ``MENACE_SAFE``.
        interval = os.getenv("AUTO_PROVISION_INTERVAL")
        if interval:
            try:
                sec = float(interval)
            except ValueError:
                sec = 0.0
            if sec > 0:
                t = threading.Thread(
                    target=self.bootstrapper.run_continuous,
                    kwargs={"interval": sec},
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
        hosts = os.getenv("REMOTE_HOSTS", "").split(",")
        hosts = [h.strip() for h in hosts if h.strip()]
        if hosts:
            self.deploy_across_hosts(hosts)
        start_scheduler_from_env()
        self.bootstrap_vector_assets()


__all__ = ["EnvironmentBootstrapper"]
