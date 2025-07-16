from __future__ import annotations

"""Helpers for discovering and generating configuration values."""

import os
import secrets
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable
import threading
import logging

from . import RAISE_ERRORS

logger = logging.getLogger(__name__)

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class ConfigDiscovery:
    """Discover local configuration such as Terraform and host info."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.failure_count = 0

    TERRAFORM_PATHS = [
        "terraform",
        "infra/terraform",
        "infrastructure/terraform",
        "deployment/terraform",
    ]
    HOST_FILES = ["hosts", "hosts.txt", "cluster_hosts", "/etc/menace/hosts"]

    def _find_terraform_dir(self) -> str | None:
        for name in self.TERRAFORM_PATHS:
            path = Path(name)
            if path.is_dir() and list(path.glob("*.tf")):
                return str(path.resolve())
        return None

    def _read_hosts(self, path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def _find_hosts(self) -> list[str]:
        for name in self.HOST_FILES:
            path = Path(name)
            if path.exists():
                hosts = self._read_hosts(path)
                if hosts:
                    return hosts
        return []

    # ------------------------------------------------------------------
    def discover(self) -> None:
        """Populate environment variables from common config locations."""

        if not os.getenv("TERRAFORM_DIR"):
            tf_dir = self._find_terraform_dir()
            if tf_dir:
                os.environ["TERRAFORM_DIR"] = tf_dir

        hosts = []
        if not os.getenv("CLUSTER_HOSTS") or not os.getenv("REMOTE_HOSTS"):
            hosts = self._find_hosts()
        if hosts:
            os.environ.setdefault("CLUSTER_HOSTS", ",".join(hosts))
            os.environ.setdefault("REMOTE_HOSTS", ",".join(hosts))

        self._detect_hardware()
        self._detect_cloud()

    # ------------------------------------------------------------------
    def _detect_hardware(self) -> None:
        """Set GPU related environment variables when possible."""
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
                gpus = len([l for l in out.splitlines() if l.strip()])
            except Exception:
                gpus = 0
        else:
            gpus = 0
        os.environ.setdefault("NUM_GPUS", str(gpus))
        os.environ.setdefault("GPU_AVAILABLE", "1" if gpus > 0 else "0")

    # ------------------------------------------------------------------
    def _detect_cloud(self) -> None:
        """Identify cloud provider and export CPU and memory info."""

        def _probe(url: str, headers: dict[str, str] | None = None) -> bool:
            try:
                req = urllib.request.Request(url, headers=headers or {})
                with urllib.request.urlopen(req, timeout=0.2):
                    return True
            except Exception:
                return False

        provider = os.getenv("CLOUD_PROVIDER")
        if not provider:
            if os.getenv("AWS_EXECUTION_ENV") or _probe("http://169.254.169.254/latest/meta-data/"):
                provider = "AWS"
            elif os.getenv("GOOGLE_CLOUD_PROJECT") or _probe("http://metadata.google.internal"):
                provider = "GCP"
            elif os.getenv("MSI_ENDPOINT") or _probe(
                "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                {"Metadata": "true"},
            ):
                provider = "AZURE"

        if provider:
            os.environ.setdefault("CLOUD_PROVIDER", provider)

        os.environ.setdefault("TOTAL_CPUS", str(os.cpu_count() or 0))
        if psutil:
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
            os.environ.setdefault("TOTAL_MEMORY_GB", str(round(mem_gb, 2)))

    # ------------------------------------------------------------------
    def run_continuous(
        self, interval: float = 3600.0, *, stop_event: threading.Event | None = None
    ) -> threading.Thread:
        """Run :meth:`discover` repeatedly in a thread."""

        if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
            return self._thread
        self._stop = stop_event or threading.Event()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.discover()
                except Exception as exc:
                    self.logger.exception("config discovery failed: %s", exc)
                    self.failure_count += 1
                if self._stop.wait(interval):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self._thread

_DEFAULT_VARS = [
    "STRIPE_API_KEY",
    "DATABASE_URL",
    "OPENAI_API_KEY",
]


def _generate_value(name: str) -> str:
    if name.endswith("_URL"):
        return f"sqlite:///{name.lower()}.db"
    return secrets.token_hex(16)


def ensure_config(vars: Iterable[str] | None = None, *, save_path: str = ".env.auto") -> None:
    required = vars or _DEFAULT_VARS
    missing = {v: _generate_value(v) for v in required if not os.getenv(v)}
    if not missing:
        return
    for k, v in missing.items():
        os.environ[k] = v
    try:
        with open(save_path, "a", encoding="utf-8") as fh:
            for k, v in missing.items():
                fh.write(f"{k}={v}\n")
    except Exception as exc:
        logger.exception("Failed to save config to %s: %s", save_path, exc)
        if RAISE_ERRORS:
            raise


__all__ = ["ensure_config", "ConfigDiscovery"]
