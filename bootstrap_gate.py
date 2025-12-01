"""Centralised readiness gate helpers for bootstrap-sensitive modules."""

from __future__ import annotations

import logging
from typing import Any, Callable, Tuple

from bootstrap_manager import bootstrap_manager
from coding_bot_interface import (
    _bootstrap_dependency_broker,
    get_active_bootstrap_pipeline,
    read_bootstrap_heartbeat,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 12.0


def _pipeline_ready_probe() -> bool:
    """Return ``True`` when an active pipeline or heartbeat is detected."""

    try:
        pipeline, manager = get_active_bootstrap_pipeline()
        if pipeline is not None or manager is not None:
            return True
    except Exception:  # pragma: no cover - best effort probe
        LOGGER.debug("bootstrap gate pipeline probe failed", exc_info=True)

    try:
        broker = _bootstrap_dependency_broker()
        resolver: Callable[[], Tuple[Any, Any]] | None = getattr(broker, "resolve", None)
        if resolver is not None:
            pipeline, manager = resolver()
            if pipeline is not None or manager is not None:
                return True
    except Exception:  # pragma: no cover - best effort probe
        LOGGER.debug("bootstrap gate broker probe failed", exc_info=True)

    try:
        return bool(read_bootstrap_heartbeat())
    except Exception:  # pragma: no cover - heartbeat is best-effort
        LOGGER.debug("bootstrap gate heartbeat probe failed", exc_info=True)
        return False


def wait_for_bootstrap_gate(
    *, timeout: float = _DEFAULT_TIMEOUT_SECONDS, description: str = "bootstrap gate"
) -> None:
    """Block until the bootstrap pipeline is ready or the gate times out."""

    try:
        bootstrap_manager.wait_until_ready(
            timeout=timeout,
            check=_pipeline_ready_probe,
            description=description,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"{description} unreachable after {timeout:.1f}s; bootstrap pipeline not ready"
        ) from exc


def resolve_bootstrap_placeholders(
    *, timeout: float = _DEFAULT_TIMEOUT_SECONDS, description: str = "bootstrap gate"
) -> tuple[Any, Any, Any]:
    """Wait for readiness and return pipeline, manager and broker placeholders."""

    wait_for_bootstrap_gate(timeout=timeout, description=description)
    pipeline, manager = get_active_bootstrap_pipeline()
    broker = _bootstrap_dependency_broker()
    return pipeline, manager, broker


__all__ = [
    "resolve_bootstrap_placeholders",
    "wait_for_bootstrap_gate",
]
