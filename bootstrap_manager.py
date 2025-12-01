"""Shared bootstrap coordination utilities.

A lightweight gate tracks in-progress and completed bootstrap helpers so that
re-entrant callers reuse the prior result instead of re-running heavy setup.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class _BootstrapStep(Generic[T]):
    def __init__(self) -> None:
        self.status: str = "pending"
        self.result: Optional[T] = None
        self.event = threading.Event()


class BootstrapManager:
    """Coordinate bootstrap helpers across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: Dict[Tuple[str, str | None], _BootstrapStep[object]] = {}
        self._ready_event = threading.Event()
        self._ready_state: Optional[bool] = None
        self._ready_error: Optional[str] = None

    def run_once(
        self,
        step: str,
        func: Callable[[], T],
        *,
        logger: logging.Logger | None = None,
        fingerprint: object | None = None,
    ) -> T:
        """Execute ``func`` at most once for the given ``step`` key.

        Additional calls while the step is running will wait for the first
        invocation to complete. Subsequent calls return the cached result
        immediately. The ``fingerprint`` parameter allows callers to include
        configuration details in the cache key without worrying about
        hashability—the manager normalises it to a ``repr`` string.
        """

        log = logger or logging.getLogger(__name__)
        fingerprint_key = repr(fingerprint) if fingerprint is not None else None
        key = (step, fingerprint_key)

        with self._lock:
            existing = self._steps.get(key)
            if existing and existing.status == "completed":
                log.info(
                    "bootstrap helper skipped (cached)",
                    extra={"bootstrap_step": step, "fingerprint": fingerprint_key, "status": "cached"},
                )
                return existing.result  # type: ignore[return-value]
            if existing and existing.status == "running":
                log.info(
                    "bootstrap helper already running",  # avoid log spam while providing context
                    extra={"bootstrap_step": step, "fingerprint": fingerprint_key, "status": "running"},
                )
                waiter = existing.event
            else:
                entry = existing or _BootstrapStep()
                entry.status = "running"
                entry.event.clear()
                self._steps[key] = entry
                waiter = None

        if waiter:
            waiter.wait()
            with self._lock:
                cached = self._steps.get(key)
                if cached and cached.status == "completed":
                    return cached.result  # type: ignore[return-value]
            return func()  # fall back to executing if the first attempt failed

        log.info(
            "bootstrap helper starting",
            extra={"bootstrap_step": step, "fingerprint": fingerprint_key, "status": "starting"},
        )

        try:
            result = func()
        except Exception:
            with self._lock:
                failure = self._steps.get(key)
                if failure:
                    failure.status = "pending"
                    failure.event.set()
            log.exception(
                "bootstrap helper failed",
                extra={"bootstrap_step": step, "fingerprint": fingerprint_key, "status": "failed"},
            )
            raise

        with self._lock:
            completed = self._steps.get(key)
            if completed:
                completed.result = result
                completed.status = "completed"
                completed.event.set()

        log.info(
            "bootstrap helper finished",
            extra={"bootstrap_step": step, "fingerprint": fingerprint_key, "status": "finished"},
        )
        return result

    def mark_ready(self, *, ready: bool = True, error: str | None = None) -> None:
        """Publish bootstrap readiness state for gated callers.

        The readiness gate is shared by all modules that need to defer work
        until the bootstrap pipeline has been prepared by an orchestrator.
        """

        with self._lock:
            self._ready_state = bool(ready)
            self._ready_error = error
            self._ready_event.set()

    def wait_until_ready(
        self,
        *,
        timeout: float | None = 10.0,
        check: Callable[[], bool] | None = None,
        poll_interval: float = 0.2,
        description: str | None = None,
    ) -> bool:
        """Wait for bootstrap readiness without re-entering bootstrap logic.

        A best-effort ``check`` callback may be provided to poll a central gate
        (for example the dependency broker) while waiting.  Calls always
        respect ``timeout`` to avoid deadlocks when the readiness gate is
        unreachable.
        """

        desc = description or "bootstrap readiness"
        if timeout is not None and timeout <= 0:
            raise TimeoutError(f"Timed out immediately waiting for {desc}; timeout must be positive")

        if self._ready_event.is_set():
            if self._ready_state:
                return True
            raise RuntimeError(self._ready_error or f"{desc} gate reported failure")

        deadline = None if timeout is None else time.monotonic() + timeout
        last_error: Exception | None = None

        while True:
            if check is not None:
                try:
                    if check():
                        self.mark_ready()
                        return True
                except Exception as exc:  # pragma: no cover - defensive guard
                    last_error = exc

            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                break

            waited = self._ready_event.wait(poll_interval if remaining is None else min(poll_interval, remaining))
            if waited:
                if self._ready_state:
                    return True
                raise RuntimeError(self._ready_error or f"{desc} gate reported failure")

        message = self._ready_error or f"Timed out waiting for {desc}"
        if last_error is not None:
            raise TimeoutError(message) from last_error
        raise TimeoutError(message)


bootstrap_manager = BootstrapManager()

__all__ = ["BootstrapManager", "bootstrap_manager"]
