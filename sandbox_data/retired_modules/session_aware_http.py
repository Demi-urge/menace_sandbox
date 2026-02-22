from __future__ import annotations

"""Session aware HTTP wrapper that auto-rotates identities."""

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
    import logging
    logging.getLogger(__name__).warning(
        "requests library missing, SessionAwareHTTP will not function: %s", exc
    )
import logging

logger = logging.getLogger(__name__)
from typing import Any, Optional
from time import sleep

from .retry_utils import retry

from .allocator_service import app as _app  # for type hints
from .session_vault import SessionData


class SessionAwareHTTP:
    """Inject cookies and user-agent from allocator service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        report_attempts: int = 3,
        report_backoff: float = 1.0,
    ) -> None:
        if requests is None:
            raise ImportError(
                "SessionAwareHTTP requires the requests package to be installed"
            )
        self.base_url = base_url.rstrip("/")
        self.current: Optional[SessionData] = None
        self.report_attempts = report_attempts
        self.report_backoff = report_backoff

    def _request_new_session(self, domain: str) -> None:
        @retry(Exception, attempts=3)
        def _post() -> object:
            return requests.post(
                f"{self.base_url}/session/get", json={"domain": domain}
            )

        resp = _post()
        resp.raise_for_status()
        data = resp.json()
        self.current = SessionData(
            cookies=data.get("cookies", {}),
            user_agent=data.get("user_agent", ""),
            fingerprint=data.get("fingerprint", ""),
            last_seen=0,
            session_id=data.get("session_id"),
            domain=domain,
        )

    def _report(self, status: str) -> None:
        if not self.current:
            return

        @retry(Exception, attempts=self.report_attempts, delay=self.report_backoff)
        def _post() -> object:
            return requests.post(
                f"{self.base_url}/session/report",
                json={"session_id": self.current.session_id, "status": status},
                timeout=5,
            )

        try:
            _post()
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("session report failed after retries: %s", exc)
            # dropping the session ensures a new one is fetched next time
            self.current = None
            raise RuntimeError(f"session report failed: {status}") from exc

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        domain = url.split("//")[-1].split("/")[0].split(":")[0]
        if not self.current or self.current.domain != domain:
            self._request_new_session(domain)
        headers = kwargs.pop("headers", {})
        headers["User-Agent"] = self.current.user_agent if self.current else ""
        cookies = kwargs.pop("cookies", {})
        if self.current:
            cookies.update(self.current.cookies)
        attempts = 0
        @retry(Exception, attempts=3)
        def _do() -> requests.Response:
            return requests.request(method, url, headers=headers, cookies=cookies, **kwargs)

        while True:
            try:
                resp = _do()
            except Exception as exc:
                self._report("error")
                raise RuntimeError(f"request to {url} failed after retries") from exc
            if resp.status_code < 400 and "captcha" not in resp.text.lower():
                self._report("success")
                return resp
            self._report("captcha" if "captcha" in resp.text.lower() else "banned")
            attempts += 1
            if attempts >= 2:
                raise RuntimeError(
                    f"request to {url} failed with status {resp.status_code}"
                )
            self._request_new_session(domain)
            sleep(0.5)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

__all__ = ["SessionAwareHTTP"]
