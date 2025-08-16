from __future__ import annotations

"""Background service for running the :class:`RelevancyRadar` periodically."""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Iterable

from .relevancy_radar import RelevancyRadar


class RelevancyRadarService:
    """Periodically scan for unused modules and trigger retirements."""

    def __init__(self, repo_root: str | Path = ".", interval: int = 3600) -> None:
        self.root = Path(repo_root)
        self.interval = interval
        self.logger = logging.getLogger(self.__class__.__name__)
        self._thread: threading.Thread | None = None
        self.running = False
        self.latest_flags: Dict[str, str] = {}
        self._retirement_service: ModuleRetirementService | None = None

    # ------------------------------------------------------------------
    def _modules(self) -> Iterable[str]:
        """Return an iterable of module identifiers under ``repo_root``."""
        try:
            from .dynamic_module_mapper import build_module_map

            mapping = build_module_map(self.root)
            return mapping
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("module map build failed")
            return []

    def _scan_once(self) -> None:
        try:
            modules = self._modules()
            flags = RelevancyRadar.flag_unused_modules(modules)
            self.latest_flags = flags
            if flags:
                if self._retirement_service is None:
                    from .module_retirement_service import ModuleRetirementService

                    self._retirement_service = ModuleRetirementService(self.root)
                self._retirement_service.process_flags(flags)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("relevancy scan failed")

    def _loop(self) -> None:
        while self.running:
            time.sleep(self.interval)
            self._scan_once()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        # Run an immediate scan so the service has up-to-date data before the
        # background thread kicks in.
        self.logger.info("performing initial relevancy scan")
        self._scan_once()
        self.logger.info("initial scan produced %d flags", len(self.latest_flags))

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None

    def flags(self) -> Dict[str, str]:
        """Return the most recent set of relevancy flags."""
        return dict(self.latest_flags)


__all__ = ["RelevancyRadarService"]
