"""Centralised readiness gate helpers for bootstrap-sensitive modules."""

# Developer note: the gate sits on top of ``bootstrap_manager`` to enforce a
# single-flight lock and idempotent reuse of the advertised pipeline/manager
# placeholders. New services should *not* bypass the gate or instantiate their
# own sentinels; instead, advertise a placeholder via
# ``coding_bot_interface.advertise_bootstrap_placeholder`` during import and
# wait on ``wait_for_bootstrap_gate``/``resolve_bootstrap_placeholders`` before
# wiring dependencies. This keeps queueing, backoff, and recursion protection in
# one place so bootstrap remains serialised and resumable across processes.

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Tuple

from bootstrap_manager import bootstrap_manager
from coding_bot_interface import (
    _bootstrap_dependency_broker,
    get_active_bootstrap_pipeline,
    read_bootstrap_heartbeat,
)
from bootstrap_timeout_policy import compute_gate_backoff, resolve_bootstrap_gate_timeout

LOGGER = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = None


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
    *, timeout: float | None = _DEFAULT_TIMEOUT_SECONDS, description: str = "bootstrap gate"
) -> None:
    """Block until the bootstrap pipeline is ready or the gate times out."""

    resolved_timeout = resolve_bootstrap_gate_timeout(fallback_timeout=timeout or 0.0)
    deadline = None if resolved_timeout is None else time.monotonic() + resolved_timeout
    attempts = 0

    while True:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            bootstrap_manager.wait_until_ready(
                timeout=remaining,
                check=_pipeline_ready_probe,
                description=description,
            )
            return
        except TimeoutError as exc:
            heartbeat = read_bootstrap_heartbeat()
            if heartbeat is None or (remaining is not None and remaining <= 0):
                raise RuntimeError(
                    f"{description} unreachable after {resolved_timeout:.1f}s; bootstrap pipeline not ready"
                ) from exc

            attempts += 1
            queue_depth = 0
            try:
                queue_depth = int(heartbeat.get("queue_depth", 0) or 0)
            except Exception:  # pragma: no cover - best effort parsing
                queue_depth = 0

            reentrant = heartbeat.get("pid") == os.getpid()
            delay = compute_gate_backoff(
                queue_depth=queue_depth,
                attempt=attempts,
                remaining=remaining,
                reentrant=reentrant,
            )
            LOGGER.info(
                "bootstrap gate busy; retrying after backoff",
                extra={
                    "event": "bootstrap-gate-backoff",
                    "queue_depth": queue_depth,
                    "attempt": attempts,
                    "delay": round(delay, 3),
                    "remaining": remaining,
                    "reentrant": reentrant,
                },
            )
            time.sleep(delay)


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
