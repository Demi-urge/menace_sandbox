"""Minimal stub of :mod:`requests` for dependency-light tests."""
from __future__ import annotations

from typing import Any


class Response:
    status_code = 0
    text = ""

    def json(self) -> Any:
        return {}


def get(*_args: Any, **_kwargs: Any) -> Response:
    raise RuntimeError("requests is unavailable in this environment")


def post(*_args: Any, **_kwargs: Any) -> Response:
    raise RuntimeError("requests is unavailable in this environment")


class Session:
    def request(self, *_args: Any, **_kwargs: Any) -> Response:
        raise RuntimeError("requests is unavailable in this environment")
