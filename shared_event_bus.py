from __future__ import annotations

"""Provide a single shared :class:`UnifiedEventBus` instance.

This module exposes ``event_bus`` which should be imported by components that
need to participate in the global publish/subscribe system.  Using a shared
instance ensures that bots across the codebase can communicate through a common
channel without each creating their own bus.
"""

try:  # pragma: no cover - allow package and flat layouts
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - flat layout fallback
    from unified_event_bus import UnifiedEventBus  # type: ignore

# Single global event bus used throughout the project.
event_bus = UnifiedEventBus()

__all__ = ["event_bus"]
