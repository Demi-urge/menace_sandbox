from __future__ import annotations

"""Automated Terraform provisioning helper."""

import logging
import os
import subprocess
from .retry_utils import retry
import threading
from typing import Optional


class InfrastructureBootstrapper:
    """Periodically apply Terraform configurations."""

    def __init__(self, tf_dir: Optional[str] = None) -> None:
        self.tf_dir = tf_dir or os.getenv("TERRAFORM_DIR")
        self.logger = logging.getLogger(self.__class__.__name__)

    def bootstrap(self) -> bool:
        if not self.tf_dir or not os.path.isdir(self.tf_dir):
            return False
        try:
            @retry(Exception, attempts=3)
            def _run(cmd: list[str]) -> subprocess.CompletedProcess:
                return subprocess.run(cmd, cwd=self.tf_dir, check=True)

            _run(["terraform", "init"])
            _run(["terraform", "apply", "-auto-approve"])
            return True
        except Exception as exc:
            self.logger.error("terraform failed: %s", exc)
            return False

    def run_continuous(self, interval: float = 86400.0, stop_event: Optional[threading.Event] = None) -> None:
        stop = stop_event or threading.Event()

        def _loop() -> None:
            while not stop.is_set():
                self.bootstrap()
                if stop.wait(interval):
                    break

        threading.Thread(target=_loop, daemon=True).start()


__all__ = ["InfrastructureBootstrapper"]
