from __future__ import annotations
"""Central configuration and secret store for Menace."""

import os
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional
import logging

from .default_config_manager import DefaultConfigManager
from .vault_secret_provider import VaultSecretProvider
from .secrets_manager import SecretsManager
from . import config_discovery as cd

logger = logging.getLogger(__name__)


class UnifiedConfigStore:
    """Load configuration from env files, Vault and secrets manager."""

    def __init__(
        self,
        path: str = ".env",
        *,
        rotation_days: int = 7,
        refresh_interval: float = 3600.0,
    ) -> None:
        self.path = Path(path)
        self.rotation_days = rotation_days
        self.refresh_interval = refresh_interval
        self.manager = SecretsManager(rotation_days=rotation_days)
        self.vault = VaultSecretProvider()
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load configuration values and export to ``os.environ``."""
        DefaultConfigManager(str(self.path)).apply_defaults()
        cd.ensure_config(save_path=str(self.path))
        if self.vault:
            for name in cd._DEFAULT_VARS:
                try:
                    self.vault.export_env(name)
                except Exception as exc:
                    logger.warning("failed to export %s: %s", name, exc)
        for line in self.path.read_text().splitlines():
            if line.strip() and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        for name in cd._DEFAULT_VARS:
            try:
                self.vault.export_env(name)
            except Exception as exc:
                logger.warning("failed to export %s: %s", name, exc)

    def start_auto_refresh(self) -> None:
        """Periodically refresh secrets from the vault."""
        if self._thread and self._thread.is_alive():
            return
        self._stop = threading.Event()

        def _loop() -> None:
            while not self._stop.is_set():
                self._refresh()
                if self._stop.wait(self.refresh_interval):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_auto_refresh(self) -> None:
        if self._stop:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


__all__ = ["UnifiedConfigStore"]
