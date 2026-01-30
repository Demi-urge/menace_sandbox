"""Minimal requests-compatible client built on urllib."""

from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request


class RequestException(Exception):
    """Base exception for request failures."""


@dataclass
class Response:
    """Lightweight response container mimicking requests.Response."""

    status_code: int
    text: str
    headers: dict[str, str]
    url: str

    def json(self) -> object:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RequestException(
                f"{self.status_code} error for url {self.url}"
            )


class Session:
    """Minimal session supporting JSON POST requests."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: object | None = None,
        timeout: float | None = None,
    ) -> Response:
        data = b""
        if json is not None:
            data = json_module.dumps(json).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return Response(
                    status_code=resp.status,
                    text=body,
                    headers=dict(resp.headers),
                    url=url,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            return Response(
                status_code=exc.code,
                text=body,
                headers=dict(exc.headers or {}),
                url=url,
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RequestException(str(exc)) from exc


json_module = json

