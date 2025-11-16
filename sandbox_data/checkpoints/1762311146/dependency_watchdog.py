from __future__ import annotations

"""Monitor external dependencies and apply remediation."""

import logging
import time
from typing import Iterable

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

from .external_dependency_provisioner import ExternalDependencyProvisioner


class DependencyWatchdog:
    """Check URLs and switch to backups if needed."""

    def __init__(
        self,
        endpoints: dict[str, str],
        backups: dict[str, str],
        *,
        attempts: int = 3,
        delay: float = 1.0,
        auto_restart: bool = False,
    ) -> None:
        self.endpoints = endpoints
        self.backups = backups
        self.attempts = attempts
        self.delay = delay
        self.auto_restart = auto_restart
        self.logger = logging.getLogger(self.__class__.__name__)
        self._provisioner: ExternalDependencyProvisioner | None = None

    # ------------------------------------------------------------------
    def _probe(self, url: str) -> bool:
        """Check a single URL with retries and exponential backoff."""
        if requests is None:
            self.logger.error("requests library missing, cannot probe %s", url)
            return False
        backoff = self.delay
        for attempt in range(1, self.attempts + 1):
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return True
                raise RuntimeError(r.status_code)
            except Exception as exc:
                self.logger.error(
                    "check failed for %s (attempt %s/%s): %s",
                    url,
                    attempt,
                    self.attempts,
                    exc,
                )
                if attempt == self.attempts:
                    break
                time.sleep(backoff)
                backoff *= 2
        return False

    # ------------------------------------------------------------------
    def check(self) -> None:
        for name, url in list(self.endpoints.items()):
            healthy = self._probe(url)
            if healthy:
                continue
            self.logger.warning("endpoint %s unhealthy", name)
            backup = self.backups.get(name)
            if backup:
                self.endpoints[name] = backup
                self.logger.info("switched %s to backup %s", name, backup)
            if self.auto_restart:
                if self._provisioner is None:
                    self._provisioner = ExternalDependencyProvisioner()
                try:
                    self.logger.info("attempting restart for %s", name)
                    self._provisioner.provision()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error("restart failed for %s: %s", name, exc)


__all__ = ["DependencyWatchdog"]
