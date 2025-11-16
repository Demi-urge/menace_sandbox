"""Confidence-based auto-override policy manager."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .stats import wilson_score_interval
from .override_db import OverrideDB
from .retry_utils import retry

if TYPE_CHECKING:
    from .chatgpt_enhancement_bot import EnhancementDB


class OverridePolicyManager:
    """Compute confidence intervals and update override flags."""

    def __init__(
        self,
        override_db: OverrideDB,
        enhancement_db: "EnhancementDB",
        window_size: int = 50,
        confidence: float = 0.95,
    ) -> None:
        self.override_db = override_db
        self.enh_db = enhancement_db
        self.window_size = window_size
        self.confidence = confidence
        self.logger = logging.getLogger(self.__class__.__name__)

    def _fetch_results(self, signature: str) -> list[int]:
        conn = self.enh_db.router.get_connection("enhancements")
        rows = conn.execute(
            "SELECT success FROM prompt_history WHERE fix=? ORDER BY id DESC LIMIT ?",
            (signature, self.window_size),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def _signatures(self) -> set[str]:
        sigs = {rec.signature for rec in self.override_db.all()}
        conn = self.enh_db.router.get_connection("enhancements")
        rows = conn.execute(
            "SELECT DISTINCT fix FROM prompt_history WHERE fix != ''"
        ).fetchall()
        sigs.update(r[0] for r in rows)
        return sigs

    @retry(Exception, attempts=3, delay=0.1)
    def update_policy(self, signature: str) -> None:
        results = self._fetch_results(signature)
        n = len(results)
        if n == 0:
            return
        successes = sum(results)
        lower, _ = wilson_score_interval(successes, n, self.confidence)
        require = lower < self.confidence
        self.override_db.set(signature, require)

    def update_all(self) -> None:
        for sig in self._signatures():
            try:
                self.update_policy(sig)
            except Exception:
                self.logger.exception("failed updating policy for %s", sig)

    def run_continuous(
        self, interval: float = 600.0, *, stop_event: threading.Event | None = None
    ) -> threading.Thread:
        if stop_event is None:
            stop_event = threading.Event()

        failure_count = 0

        def _loop() -> None:
            nonlocal failure_count
            while not stop_event.is_set():
                try:
                    self.update_all()
                    failure_count = 0
                except Exception:
                    failure_count += 1
                    self.logger.exception("policy update cycle failed")
                    if failure_count >= 3:
                        self.logger.error(
                            "policy updates failing repeatedly (%s times)",
                            failure_count,
                        )
                if stop_event.wait(interval):
                    break

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        return thread

    def require_human(self, signature: str) -> bool:
        rec = self.override_db.get(signature)
        if rec is None:
            self.update_policy(signature)
            rec = self.override_db.get(signature)
        return True if rec is None else rec.require_human

    def record_fix(self, signature: str, success: bool) -> None:
        self.update_policy(signature)


__all__ = ["OverridePolicyManager", "OverrideDB"]
