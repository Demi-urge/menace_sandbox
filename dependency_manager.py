from __future__ import annotations

"""Optional dependency loader registry."""

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class DependencyManager:
    """Manage optional imports and expose loaded modules."""

    def __init__(self) -> None:
        self._deps: dict[str, Any] = {}
        self._errors: dict[str, Exception] = {}

    def load(self, name: str, loader: Callable[[], Any]) -> Any | None:
        """Attempt to load a dependency via ``loader`` and cache the result."""
        if name in self._deps:
            return self._deps[name]
        try:
            mod = loader()
        except Exception as exc:  # pragma: no cover - optional
            logger.debug("optional dependency %s unavailable: %s", name, exc)
            self._deps[name] = None
            self._errors[name] = exc
            return None
        else:
            self._deps[name] = mod
            return mod

    def get(self, name: str) -> Any | None:
        return self._deps.get(name)

    def error(self, name: str) -> Exception | None:
        return self._errors.get(name)


__all__ = ["DependencyManager"]
