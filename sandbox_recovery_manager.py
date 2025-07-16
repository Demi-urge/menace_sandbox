from __future__ import annotations

"""Restart sandbox runs when unexpected failures occur."""

from typing import Any, Callable, Dict
import argparse
import logging
import time

logger = logging.getLogger(__name__)


class SandboxRecoveryManager:
    """Wrap ``_sandbox_main`` and restart on uncaught errors."""

    def __init__(
        self,
        sandbox_main: Callable[[Dict[str, Any], argparse.Namespace], Any],
        *,
        retry_delay: float = 1.0,
        max_retries: int | None = None,
    ) -> None:
        self.sandbox_main = sandbox_main
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def run(self, preset: Dict[str, Any], args: argparse.Namespace):
        """Execute ``sandbox_main`` retrying on failure."""
        attempts = 0
        while True:
            try:
                return self.sandbox_main(preset, args)
            except Exception:  # pragma: no cover - rare
                attempts += 1
                self.logger.exception("sandbox run crashed; restarting")
                if self.max_retries is not None and attempts >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)


__all__ = ["SandboxRecoveryManager"]
