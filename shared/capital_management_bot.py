"""Lightweight accessors for :mod:`menace_sandbox.capital_management_bot`.

This shim mirrors the lazy import pattern used by
``menace_sandbox.shared.model_pipeline_core`` so that shared modules can depend
on :class:`CapitalManagementBot` without importing the full capital management
stack during bootstrap.  Keeping the wrapper in ``shared`` allows neutral
modules such as :mod:`menace_sandbox.shared.pipeline_base` to reference the
capital manager for type checking while avoiding circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only import
    from ..capital_management_bot import CapitalManagementBot as _CapitalManagementBot
else:  # pragma: no cover - runtime fallback avoids circular imports
    _CapitalManagementBot = Any  # type: ignore[misc, assignment]

_CAPITAL_MANAGER: "type[_CapitalManagementBot] | None" = None

__all__ = ["CapitalManagementBot"]


def _load_capital_manager_cls() -> "type[_CapitalManagementBot]":
    """Import and cache the concrete :class:`CapitalManagementBot` implementation."""

    global _CAPITAL_MANAGER
    if _CAPITAL_MANAGER is None:
        from ..capital_management_bot import CapitalManagementBot as _CapitalManager

        _CAPITAL_MANAGER = _CapitalManager
        globals()["CapitalManagementBot"] = _CAPITAL_MANAGER
    return _CAPITAL_MANAGER


def __getattr__(name: str) -> Any:
    """Provide lazy access to :class:`CapitalManagementBot`."""

    if name == "CapitalManagementBot":
        return _load_capital_manager_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazily provided attributes to :func:`dir`."""

    return sorted(list(globals().keys()) + ["CapitalManagementBot"])

