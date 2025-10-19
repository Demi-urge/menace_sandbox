"""Minimal stub of :mod:`pydantic` providing core symbols for tests."""
from __future__ import annotations

from typing import Any, Callable


class BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self, *_, **__) -> dict[str, Any]:
        return self.__dict__.copy()


def Field(default: Any = None, **_: Any) -> Any:  # noqa: D401 - stub helper
    return default


class ValidationError(Exception):
    pass


def validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap


def root_validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap
