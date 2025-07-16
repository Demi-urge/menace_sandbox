"""Client for menace_visual_agent_2 service."""

from __future__ import annotations

import os
import subprocess
import time
import logging
from typing import Iterable, Dict, Any

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

SELF_IMPROVEMENT_PREFIX = (
    "Improve Menace by enhancing error handling and modifying existing bots."
)


class VisualAgentClient:
    """Interface used by SelfCodingEngine to contact a visual agent."""

    def __init__(
        self,
        urls: Iterable[str] | None = None,
        token: str | None = None,
        poll_interval: float | None = None,
        token_refresh_cmd: str | None = None,
    ) -> None:
        default_urls = (
            os.getenv("VISUAL_AGENT_URLS")
            or os.getenv("VISUAL_DESKTOP_URL", "http://127.0.0.1:8001")
        ).split(";")
        self.urls = list(filter(None, urls or default_urls))
        self.token = token or os.getenv("VISUAL_AGENT_TOKEN", "")
        self.poll_interval = poll_interval or float(
            os.getenv("VISUAL_AGENT_POLL_INTERVAL", "5")
        )
        self.token_refresh_cmd = token_refresh_cmd or os.getenv(
            "VISUAL_TOKEN_REFRESH_CMD"
        )

    def _poll(self, base: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"
        while True:
            try:
                resp = requests.get(f"{base}/status", timeout=10)
            except Exception as exc:  # pragma: no cover - network issues
                return False, f"status poll failed: {exc}"
            if resp.status_code != 200:
                return False, f"unexpected status {resp.status_code}"
            data = resp.json()
            if not data.get("active", False):
                return True, str(data.get("status", "done"))
            time.sleep(self.poll_interval)

    def _refresh_token(self) -> bool:
        if not self.token_refresh_cmd:
            return False
        try:
            new = subprocess.check_output(
                self.token_refresh_cmd, shell=True, text=True
            ).strip()
            if new:
                self.token = new
                return True
        except Exception:
            return False
        return False

    def _send(self, base: str, prompt: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"

        def sender() -> tuple[bool, str]:
            resp = requests.post(
                f"{base}/run",
                headers={"x-token": self.token},
                json={"prompt": prompt, "branch": None},
                timeout=10,
            )
            if resp.status_code == 401:
                if self._refresh_token():
                    return sender()
                return False, "unauthorized"
            if resp.status_code == 202:
                return self._poll(base)
            snippet = resp.text[:200]
            return False, f"status {resp.status_code}: {snippet}"

        try:
            return sender()
        except Exception as exc:  # pragma: no cover - network issues
            return False, f"exception {exc}"

    def _send_revert(self, base: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"

        def sender() -> tuple[bool, str]:
            resp = requests.post(
                f"{base}/revert",
                headers={"x-token": self.token},
                timeout=10,
            )
            if resp.status_code == 401:
                if self._refresh_token():
                    return sender()
                return False, "unauthorized"
            if resp.status_code == 202:
                return self._poll(base)
            snippet = resp.text[:200]
            return False, f"status {resp.status_code}: {snippet}"

        try:
            return sender()
        except Exception as exc:  # pragma: no cover - network issues
            return False, f"exception {exc}"

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        prompt = SELF_IMPROVEMENT_PREFIX + "\n\n" + "\n".join(
            m.get("content", "") for m in messages
        )
        last_reason = "failed"
        for url in self.urls:
            ok, reason = self._send(url, prompt)
            if ok:
                return {"choices": [{"message": {"content": reason}}]}
            last_reason = reason
        raise RuntimeError(f"visual agent failed: {last_reason}")

    def revert(self) -> bool:
        """Trigger a revert of the last merge via the visual agent."""
        last_reason = "failed"
        for url in self.urls:
            ok, reason = self._send_revert(url)
            if ok:
                return True
            last_reason = reason
        raise RuntimeError(f"visual agent revert failed: {last_reason}")


class VisualAgentClientStub:
    """Fallback client used when requests or the real client is unavailable."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        logger.info("VisualAgentClientStub active")
        self.urls: list[str] = []

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": ""}}]}

    def revert(self) -> bool:
        return False


__all__ = ["VisualAgentClient", "VisualAgentClientStub"]

