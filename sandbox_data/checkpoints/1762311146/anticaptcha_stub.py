"""CAPTCHA solving helper that delegates to a remote service."""

from __future__ import annotations

import logging
import os
import base64
import time
from dataclasses import dataclass
logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """Outcome of a CAPTCHA solve attempt."""

    text: str | None
    used_remote: bool
    error: str | None = None


class AntiCaptchaClient:
    """Client that attempts to solve CAPTCHA images locally or remotely."""

    def __init__(self, api_key: str | None = None, *, language: str = "eng") -> None:
        self.api_key = api_key
        self.language = language

    def solve(self, image_path: str, *, language: str | None = None) -> SolveResult:
        """Attempt to extract text from ``image_path`` using a remote service."""
        logger.debug("AntiCaptchaClient.solve called with %s", image_path)
        text, err = self._remote_with_retry(image_path)
        return SolveResult(text, bool(self.api_key), err)

    def _remote_with_retry(self, image_path: str) -> tuple[str | None, str | None]:
        """Attempt remote solving with retries and report the last error."""

        attempts = int(os.getenv("ANTICAPTCHA_ATTEMPTS", "3"))
        backoff = float(os.getenv("ANTICAPTCHA_BACKOFF", "1"))
        last_err: str | None = None
        for i in range(attempts):
            text, err = self._remote_solve(image_path)
            if text:
                return text, None
            last_err = err
            if i < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
        return None, last_err

    def _remote_solve(self, image_path: str) -> tuple[str | None, str | None]:
        """Send the image to an external service when local OCR fails."""

        if not self.api_key:
            logger.debug("No API key provided; skipping remote solve")
            return None, "no api key"
        try:
            import requests  # type: ignore

            with open(image_path, "rb") as fh:
                img_bytes = fh.read()

            # choose provider behaviour based on env vars
            provider = os.getenv("ANTICAPTCHA_PROVIDER", "simple")
            if provider == "2captcha":
                in_url = os.getenv("ANTICAPTCHA_IN_URL", "https://2captcha.com/in.php")
                res_url = os.getenv("ANTICAPTCHA_RES_URL", "https://2captcha.com/res.php")
                resp = requests.post(
                    in_url,
                    files={"file": img_bytes},
                    data={"key": self.api_key, "method": "post"},
                    timeout=10,
                )
                resp.raise_for_status()
                if "OK|" not in resp.text:
                    logger.error("Unexpected 2captcha response: %s", resp.text)
                    return None, resp.text
                cid = resp.text.split("|", 1)[1]
                max_polls = int(os.getenv("ANTICAPTCHA_MAX_POLLS", "20"))
                interval = float(os.getenv("ANTICAPTCHA_POLL_INTERVAL", "5"))
                for _ in range(max_polls):
                    time.sleep(interval)
                    r2 = requests.get(
                        res_url,
                        params={"key": self.api_key, "action": "get", "id": cid},
                        timeout=10,
                    )
                    r2.raise_for_status()
                    if r2.text == "CAPCHA_NOT_READY":
                        continue
                    if "OK|" in r2.text:
                        return r2.text.split("|", 1)[1].strip(), None
                    logger.error("2captcha reported error: %s", r2.text)
                    return None, r2.text
            else:
                url = os.getenv("ANTICAPTCHA_URL", "http://localhost:8002/solve")
                b64 = base64.b64encode(img_bytes).decode()
                resp = requests.post(
                    url,
                    json={"key": self.api_key, "body": b64, "language": self.language},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("text") or "").strip()
                if text:
                    return text, None
                return None, "empty response"
        except Exception as exc:  # pragma: no cover - network issues
            logger.error("Remote solve failed: %s", exc)
            return None, str(exc)
        return None, "unsolved"


__all__ = ["AntiCaptchaClient", "SolveResult"]
