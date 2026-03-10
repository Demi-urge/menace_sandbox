"""Fallback shim for optional ``SelfCodingManager`` imports."""

from __future__ import annotations

from typing import Any


class SelfCodingManagerShim:
    """Mutable no-op manager used when self-coding dependencies are unavailable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


__all__ = ["SelfCodingManagerShim"]
