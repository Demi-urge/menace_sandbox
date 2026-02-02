"""Minimal stub of :mod:`pydantic` providing core symbols for sandbox runs."""
from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "RootModel",
    "ValidationError",
    "field_validator",
    "model_validator",
    "validator",
    "root_validator",
]


class BaseModel:
    model_fields: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self, *_: Any, **__: Any) -> dict[str, Any]:
        return self.__dict__.copy()


class ConfigDict(dict):
    """Lightweight stand-in for pydantic v2 ConfigDict."""


class RootModel(BaseModel):
    """Minimal RootModel shim that stores the root value."""

    def __init__(self, root: Any) -> None:
        super().__init__(root=root)

    def __class_getitem__(cls, _item: Any) -> type["RootModel"]:
        return cls


def Field(default: Any = None, **_: Any) -> Any:  # noqa: D401 - stub helper
    return default


class ValidationError(Exception):
    pass


def validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap


def field_validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap


def model_validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap


def root_validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap
