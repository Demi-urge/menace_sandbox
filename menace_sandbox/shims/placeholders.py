from __future__ import annotations

"""Reusable placeholder shim types for optional runtime dependencies."""

from typing import Any, Callable


_SENTINEL = "<shim>"


class CallableShim:
    """Callable no-op shim with deterministic return values."""

    def __init__(self, *args: Any, return_value: Any = None, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.return_value = return_value

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.return_value

    def start(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def stop(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def run(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.return_value

    def record(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def add(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def fit(self, *_args: Any, **_kwargs: Any) -> "CallableShim":
        return self

    @classmethod
    def with_return(cls, value: Any) -> Callable[..., Any]:
        def _factory(*args: Any, **kwargs: Any) -> "CallableShim":
            return cls(*args, return_value=value, **kwargs)

        return _factory


class ManagerShim(CallableShim):
    """Manager-like shim exposing empty collection accessors."""

    def list(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def get(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def items(self, *_args: Any, **_kwargs: Any) -> list[tuple[Any, Any]]:
        return []


class NullServiceShim(ManagerShim):
    """Service shim that reports a disabled state with sentinel values."""

    sentinel = _SENTINEL

    def status(self, *_args: Any, **_kwargs: Any) -> str:
        return "disabled"

    def ping(self, *_args: Any, **_kwargs: Any) -> str:
        return self.sentinel

    def health(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": False, "service": self.sentinel}


__all__ = ["CallableShim", "ManagerShim", "NullServiceShim"]
