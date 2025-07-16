from __future__ import annotations

"""High-level CAPTCHA handling pipeline using Playwright."""

from typing import Optional
import logging
import os

from .captcha_system import CaptchaDetector, CaptchaManager
from .replay_engine import ReplayEngine


logger = logging.getLogger(__name__)


class CaptchaPipeline:
    """Coordinate detection, snapshotting and replay."""

    def __init__(self, manager: CaptchaManager,
                 detector: Optional[CaptchaDetector] = None,
                 replay: Optional[ReplayEngine] = None) -> None:
        self.manager = manager
        self.detector = detector or CaptchaDetector()
        self.replay = replay
        # testing aids
        self.last_detected: bool = False
        self.last_token: str | None = None
        self.last_error: Exception | None = None
        self.last_snapshot: str | None = None

    async def run(self, page, job_id: str, *, timeout: float | None = None) -> None:
        """Check the page for CAPTCHAs and pause until solved.

        The ``timeout`` value is forwarded to :meth:`CaptchaManager.wait_for_solution`
        so callers don't accidentally hang forever if a token never arrives.
        """
        self.last_detected = False
        self.last_token = None
        self.last_error = None
        self.last_snapshot = None

        data = None
        if self.replay:
            try:
                data = await self.replay.record(page)
                logger.debug("replay recording started for %s", job_id)
            except Exception as exc:
                self.last_error = exc
                logger.error("replay record failed: %s", exc)

        try:
            html = await page.content()
        except Exception as exc:
            self.last_error = exc
            logger.error("failed to fetch page content: %s", exc)
            return

        self.last_detected = self.detector.detect(html)
        if not self.last_detected:
            logger.debug("no CAPTCHA detected for %s", job_id)
            return
        logger.info("CAPTCHA detected for %s", job_id)

        if self.replay and data:
            try:
                await self.replay.pause(page, data)
                logger.debug("replay paused for %s", job_id)
            except Exception as exc:
                self.last_error = exc
                logger.error("replay pause failed: %s", exc)

        snapshot = None
        try:
            info = await self.manager.snapshot_and_pause(page, job_id)
            snapshot = info.key_base
            self.last_snapshot = snapshot
            logger.debug("snapshot saved for %s", job_id)
        except Exception as exc:
            self.last_error = exc
            logger.error("snapshot_and_pause failed: %s", exc)
            return

        eff_timeout = timeout
        if eff_timeout is None:
            config = getattr(self.manager, "config", None)
            if isinstance(config, dict) and config.get("captcha_timeout"):
                eff_timeout = float(config["captcha_timeout"])
            else:
                eff_timeout = float(os.getenv("CAPTCHA_TIMEOUT", "60"))

        try:
            token = await self.manager.wait_for_solution(job_id, timeout=eff_timeout)
            self.last_token = token
            logger.info("CAPTCHA solved for %s", job_id)
        except Exception as exc:
            self.last_error = exc
            logger.error("waiting for solution failed: %s", exc)
            return

        if self.replay and data:
            try:
                await self.replay.resume(page.context, data, token)
                logger.debug("replay resumed for %s", job_id)
            except Exception as exc:
                self.last_error = exc
                logger.error("replay resume failed: %s", exc)


__all__ = ["CaptchaPipeline"]
