from __future__ import annotations

"""Restart sandbox runs when unexpected failures occur."""

from typing import Any, Callable, Dict
import argparse
import logging
import os
from pathlib import Path
import time
import traceback

logger = logging.getLogger(__name__)


class SandboxRecoveryManager:
    """Wrap ``_sandbox_main`` and restart on uncaught errors."""

    def __init__(
        self,
        sandbox_main: Callable[[Dict[str, Any], argparse.Namespace], Any],
        *,
        retry_delay: float = 1.0,
        max_retries: int | None = None,
        on_retry: Callable[[Exception, float], None] | None = None,
    ) -> None:
        self.sandbox_main = sandbox_main
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
        self.on_retry = on_retry

    # ------------------------------------------------------------------
    def run(self, preset: Dict[str, Any], args: argparse.Namespace):
        """Execute ``sandbox_main`` retrying on failure."""
        attempts = 0
        while True:
            start = time.monotonic()
            try:
                return self.sandbox_main(preset, args)
            except Exception as exc:  # pragma: no cover - rare
                attempts += 1
                runtime = time.monotonic() - start
                self.logger.exception("sandbox run crashed; restarting")

                log_dir = Path(
                    getattr(args, "sandbox_data_dir", None)
                    or os.getenv("SANDBOX_DATA_DIR", "sandbox_data")
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "recovery.log"
                tb = traceback.format_exc()
                with open(log_file, "a", encoding="utf-8") as fh:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    fh.write(
                        f"{ts} attempt={attempts} runtime={runtime:.2f}s\n{tb}\n"
                    )

                if self.on_retry:
                    try:
                        self.on_retry(exc, runtime)
                    except Exception:
                        self.logger.exception("on_retry callback failed")

                if self.max_retries is not None and attempts >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)


__all__ = ["SandboxRecoveryManager"]
