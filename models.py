from __future__ import annotations

"""Data models for the access control system."""

from dataclasses import dataclass

try:  # pragma: no cover - support flat module execution
    from .roles import ROLE_PERMISSIONS, READ
except ImportError:  # pragma: no cover - fallback when package context missing
    from roles import ROLE_PERMISSIONS, READ  # type: ignore


@dataclass
class BotRole:
    """Represents a bot and its associated role."""

    name: str
    role: str = READ

    def __post_init__(self) -> None:
        if self.role not in ROLE_PERMISSIONS:
            raise ValueError(f"Invalid role: {self.role}")

__all__ = ["BotRole"]
