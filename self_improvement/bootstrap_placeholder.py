"""Shared bootstrap placeholder utilities for self-improvement entrypoints."""

from __future__ import annotations

from typing import Any

from ..coding_bot_interface import (
    _bootstrap_dependency_broker,
    advertise_bootstrap_placeholder,
)


def advertise_broker_placeholder(
    *,
    dependency_broker: Any | None = None,
    pipeline: Any | None = None,
    manager: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Advertise the bootstrap placeholder with ``owner=True``."""

    broker = dependency_broker or _bootstrap_dependency_broker()
    pipeline_placeholder, manager_placeholder = advertise_bootstrap_placeholder(
        dependency_broker=broker,
        pipeline=pipeline,
        manager=manager,
        owner=True,
    )
    return pipeline_placeholder, manager_placeholder, broker


def bootstrap_broker() -> Any:
    """Expose the active bootstrap dependency broker for reuse."""

    return _bootstrap_dependency_broker()


__all__ = [
    "advertise_broker_placeholder",
    "bootstrap_broker",
]
