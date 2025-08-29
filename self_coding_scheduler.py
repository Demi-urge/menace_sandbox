from __future__ import annotations

"""Scheduler running self-coding when metrics degrade."""

import threading
import time
import logging
from pathlib import Path
from typing import Iterable, Optional, List

from .self_coding_manager import SelfCodingManager
from .data_bot import DataBot
from .advanced_error_management import AutomatedRollbackManager
from .sandbox_settings import SandboxSettings


class SelfCodingScheduler:
    """Trigger :class:`SelfCodingManager` based on ROI and error metrics.

    Defaults for ``interval``, ``roi_drop`` and ``error_increase`` are loaded
    from :class:`SandboxSettings` (``SELF_CODING_INTERVAL``,
    ``SELF_CODING_ROI_DROP`` and ``SELF_CODING_ERROR_INCREASE``) and can be
    overridden via constructor arguments.
    """

    def __init__(
        self,
        manager: SelfCodingManager,
        data_bot: DataBot,
        *,
        rollback_mgr: AutomatedRollbackManager | None = None,
        nodes: Optional[Iterable[str]] = None,
        interval: int | None = None,
        roi_drop: float | None = None,
        error_increase: float | None = None,
        patch_path: Path | None = None,
        description: str = "auto_patch",
        settings: SandboxSettings | None = None,
    ) -> None:
        self.manager = manager
        self.data_bot = data_bot
        self.rollback_mgr = rollback_mgr
        self.nodes: List[str] = list(nodes or [])
        self.settings = settings or SandboxSettings()
        self.interval = interval if interval is not None else self.settings.self_coding_interval
        self.roi_drop = (
            roi_drop if roi_drop is not None else self.settings.self_coding_roi_drop
        )
        self.error_increase = (
            error_increase
            if error_increase is not None
            else self.settings.self_coding_error_increase
        )
        self.patch_path = patch_path or Path("auto_helpers.py")
        self.description = description
        self.last_roi = self.data_bot.roi(self.manager.bot_name)
        self.last_errors = 0.0
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def _current_errors(self) -> float:
        df = self.data_bot.db.fetch(10)
        if hasattr(df, "empty"):
            df = df[df["bot"] == self.manager.bot_name]
            if df.empty:
                return 0.0
            return float(df["errors"].mean())
        if isinstance(df, list):
            rows = [r for r in df if r.get("bot") == self.manager.bot_name]
            if not rows:
                return 0.0
            return float(sum(r.get("errors", 0.0) for r in rows) / len(rows))
        return 0.0

    def _latest_patch_id(self) -> str | None:
        patch_db = getattr(self.manager.engine, "patch_db", None)
        if not patch_db:
            return None
        try:
            row = patch_db.conn.execute(
                "SELECT id FROM patch_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except Exception:
            return None
        return str(row[0]) if row else None

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            try:
                roi = self.data_bot.roi(self.manager.bot_name)
                errors = self._current_errors()
                if (
                    roi - self.last_roi <= self.roi_drop
                    or errors - self.last_errors >= self.error_increase
                ):
                    before = roi
                    self.manager.run_patch(self.patch_path, self.description)
                    after = self.data_bot.roi(self.manager.bot_name)
                    if after < before:
                        pid = self._latest_patch_id()
                        if pid:
                            self.manager.engine.rollback_patch(pid)
                            if self.rollback_mgr and self.nodes:
                                try:
                                    self.rollback_mgr.auto_rollback(pid, self.nodes)
                                except Exception:
                                    self.logger.exception("auto rollback failed")
                    self.last_roi = after
                    self.last_errors = errors
                else:
                    self.last_roi = roi
                    self.last_errors = errors
            except Exception:
                self.logger.exception("self-coding loop failed")
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None


__all__ = ["SelfCodingScheduler"]
