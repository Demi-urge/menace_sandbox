"""Captcha handling utilities for Menace."""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import logging
import tempfile

try:  # optional dependencies
    import boto3
except Exception:  # pragma: no cover - optional
    boto3 = None  # type: ignore

try:  # optional
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore
from .anticaptcha_stub import AntiCaptchaClient
try:
    from .metrics_exporter import Gauge
except Exception:  # pragma: no cover - optional dependency
    Gauge = None  # type: ignore

_TOKEN_RE = re.compile(r"^[\w.-]+$")


def _valid_token(token: str) -> bool:
    """Return True if the token looks plausible."""
    return bool(token and _TOKEN_RE.match(token))


# ---------------------------------------------------------------------------
class CaptchaDetector:
    """Detect CAPTCHA challenges using DOM patterns."""

    def __init__(self, patterns: Optional[list[str]] = None) -> None:
        self.patterns = [re.compile(p, re.I) for p in (patterns or ["captcha"])]

    def detect(self, html: str, screenshot_path: str | None = None) -> bool:
        """Return True if a CAPTCHA challenge is likely present."""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

        # direct keyword search
        for pat in self.patterns:
            if pat.search(text):
                return True

        # common DOM features
        for attr in ["id", "class", "name"]:
            if soup.find(attrs={attr: re.compile("captcha", re.I)}):
                return True
        if soup.find("iframe", src=re.compile("(recaptcha|hcaptcha)", re.I)):
            return True
        return False


# ---------------------------------------------------------------------------
@dataclass
class SnapshotInfo:
    job_id: str
    key_base: str


class CaptchaManager:
    """Handle snapshot storage and job state transitions."""

    def __init__(
        self,
        bucket: str,
        redis_url: str,
        *,
        anticaptcha_api_key: str | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bucket = bucket
        self.s3 = boto3.client("s3") if boto3 else None
        try:
            self.redis = (
                redis.Redis.from_url(redis_url, decode_responses=True) if redis else None
            )
            if self.redis:
                self.redis.ping()
        except Exception as exc:  # pragma: no cover - no server running
            logging.warning("Redis unavailable: %s", exc)
            self.redis = None
        self.local_state: dict[str, dict[str, str]] = {}
        self.subscribers: List[asyncio.Queue] = []
        key = anticaptcha_api_key or os.getenv("ANTICAPTCHA_API_KEY")
        self.anticaptcha_client: AntiCaptchaClient | None = AntiCaptchaClient(key)
        cfg = config or {}
        self.config = {
            "remote_attempts": int(
                cfg.get("remote_attempts", os.getenv("CAPTCHA_REMOTE_ATTEMPTS", "3"))
            ),
            "remote_backoff": float(
                cfg.get("remote_backoff", os.getenv("CAPTCHA_REMOTE_BACKOFF", "1"))
            ),
            "captcha_timeout": float(
                cfg.get("captcha_timeout", os.getenv("CAPTCHA_TIMEOUT", "60"))
            ),
        }
        self.metrics: Dict[str, Gauge] = {}
        if Gauge:
            self.metrics = {
                "total_captchas_solved": Gauge(
                    "captchas_solved_total", "Total CAPTCHAs solved"
                ),
                "remote_failures": Gauge(
                    "captcha_remote_failures_total", "Remote CAPTCHA solve failures"
                ),
            }
        
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(q)
        return q

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except Exception as exc:
                logging.error("failed to notify subscriber: %s", exc)

    async def snapshot_and_pause(self, page, job_id: str) -> SnapshotInfo:
        screenshot = await page.screenshot()
        html = await page.content()
        key_base = f"{job_id}/{int(time.time())}"
        if self.s3:
            try:
                self.s3.put_object(Bucket=self.bucket, Key=f"{key_base}.png", Body=screenshot)
                self.s3.put_object(Bucket=self.bucket, Key=f"{key_base}.html", Body=html.encode())
            except Exception as exc:  # pragma: no cover - network issues
                logging.warning("failed to upload snapshot to S3: %s", exc)
                raise
        else:
            p = Path(key_base)
            p.parent.mkdir(parents=True, exist_ok=True)
            Path(f"{key_base}.png").write_bytes(screenshot)
            Path(f"{key_base}.html").write_text(html)
        mapping = {"state": "BLOCKED", "snapshot": key_base}
        if self.redis:
            try:
                self.redis.hset(job_id, mapping=mapping)
            except Exception as exc:  # pragma: no cover - network issues
                logging.warning("failed to update Redis for %s: %s", job_id, exc)
                self.local_state[job_id] = mapping
        else:
            self.local_state[job_id] = mapping
        await self._broadcast({"type": "blocked", "job_id": job_id, "snapshot": key_base})
        return SnapshotInfo(job_id=job_id, key_base=key_base)

    def mark_resolved(self, job_id: str, token: str) -> None:
        if not _valid_token(token):
            raise ValueError(f"invalid captcha token: {token!r}")
        mapping = {"state": "SOLVED", "token": token}
        if self.redis:
            try:
                self.redis.hset(job_id, mapping=mapping)
            except Exception as exc:  # pragma: no cover
                logging.warning("failed to update Redis for %s: %s", job_id, exc)
                self.local_state[job_id] = mapping
        else:
            self.local_state[job_id] = mapping
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.create_task(self._broadcast({"type": "solved", "job_id": job_id}))
        else:
            asyncio.run(self._broadcast({"type": "solved", "job_id": job_id}))
        if "total_captchas_solved" in self.metrics:
            self.metrics["total_captchas_solved"].inc()

    def _solve_snapshot(self, key_base: str) -> str | None:
        """Attempt to solve the CAPTCHA snapshot using a remote solver.

        Remote solving may be retried a few times because transient network
        issues are fairly common. The number of attempts and initial backoff
        delay can be tweaked via the environment variables
        ``CAPTCHA_REMOTE_ATTEMPTS`` and ``CAPTCHA_REMOTE_BACKOFF``.
        """
        if not self.anticaptcha_client:
            return None
        path = None
        if self.s3:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=f"{key_base}.png")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as fh:
                    fh.write(obj["Body"].read())
                    path = fh.name
            except Exception as exc:  # pragma: no cover - network issues
                logging.error("failed to download snapshot: %s", exc)
                return None
        else:
            path = f"{key_base}.png"
            if not Path(path).exists():
                return None

        token: str | None = None
        err: str | None = None

        # Try remote solver first with basic retries
        attempts = int(self.config.get("remote_attempts", 3))
        backoff = float(self.config.get("remote_backoff", 1.0))
        for i in range(attempts):
            try:
                token, err = self.anticaptcha_client._remote_solve(path)
            except Exception as exc:  # pragma: no cover - network issues
                logging.error("remote solving failed: %s", exc)
                err = str(exc)
            if token and _valid_token(token):
                break
            token = None
            time.sleep(backoff)
            backoff *= 2
        else:
            if "remote_failures" in self.metrics:
                self.metrics["remote_failures"].inc()

        return token

    async def wait_for_solution(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> str:
        """Poll Redis/local state until a CAPTCHA token is available.

        The call will raise ``TimeoutError`` when no token is obtained within
        ``timeout`` seconds. A ``None`` value disables the timeout entirely,
        falling back to the ``captcha_timeout`` config value (or the
        ``CAPTCHA_TIMEOUT`` environment variable). This ensures the coroutine
        never blocks forever when the caller forgets to specify a limit.
        """
        if timeout is None:
            timeout = float(self.config.get("captcha_timeout", 60.0))
        start = time.time()
        while True:
            record = None
            if self.redis:
                try:
                    record = self.redis.hgetall(job_id)
                except Exception as exc:
                    logging.warning("failed to fetch state from Redis for %s: %s", job_id, exc)
                    record = self.local_state.get(job_id)
            else:
                record = self.local_state.get(job_id)
            if record and record.get("state") == "SOLVED" and record.get("token"):
                return record["token"]
            if record and record.get("snapshot"):
                loop = asyncio.get_running_loop()
                token = await loop.run_in_executor(
                    None, self._solve_snapshot, record["snapshot"]
                )
                if token:
                    self.mark_resolved(job_id, token)
                    return token
            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError(f"captcha solving timed out for {job_id}")
            await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
async def solver_loop(manager: CaptchaManager, poll_interval: float = 2.0) -> None:
    """Background task that automatically solves blocked jobs."""

    q = manager.subscribe()
    while True:
        event = await q.get()
        if event.get("type") != "blocked":
            continue
        job_id = event["job_id"]

        async def _solve() -> None:
            try:
                await manager.wait_for_solution(job_id, poll_interval=poll_interval, timeout=None)
            except Exception as exc:  # pragma: no cover - runtime issues
                logging.error("solver loop failed for %s: %s", job_id, exc)

        asyncio.create_task(_solve())


__all__ = [
    "CaptchaDetector",
    "CaptchaManager",
    "SnapshotInfo",
    "solver_loop",
]
