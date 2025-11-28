"""Shared timeout policy helpers for bootstrap entry points."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from typing import Callable, Dict, Iterator, Mapping, MutableMapping

LOGGER = logging.getLogger(__name__)

_BOOTSTRAP_TIMEOUT_MINIMUMS: dict[str, float] = {
    "MENACE_BOOTSTRAP_WAIT_SECS": 240.0,
    "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": 240.0,
    "BOOTSTRAP_STEP_TIMEOUT": 240.0,
    "BOOTSTRAP_VECTOR_STEP_TIMEOUT": 240.0,
}
_OVERRIDE_ENV = "MENACE_BOOTSTRAP_TIMEOUT_ALLOW_UNSAFE"


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enforce_bootstrap_timeout_policy(
    *,
    logger: logging.Logger | None = None,
    prompt_override: Callable[[str, float, float], bool] | None = None,
) -> Dict[str, Dict[str, float | bool | None]]:
    """Clamp bootstrap timeouts to recommended floors when needed."""

    active_logger = logger or LOGGER
    allow_unsafe = _truthy_env(os.getenv(_OVERRIDE_ENV))
    results: Dict[str, Dict[str, float | bool | None]] = {}

    for env_var, minimum in _BOOTSTRAP_TIMEOUT_MINIMUMS.items():
        raw_value = os.getenv(env_var)
        requested_value = _parse_float(raw_value)
        effective_value = requested_value if requested_value is not None else minimum
        clamped = False
        override_granted = False

        if requested_value is None and raw_value is not None:
            active_logger.warning(
                "%s is not a valid float (%r); forcing recommended minimum %.1fs",
                env_var,
                raw_value,
                minimum,
                extra={"env_var": env_var, "raw_value": raw_value, "minimum": minimum},
            )
            clamped = True
            effective_value = minimum
            os.environ[env_var] = str(minimum)
        elif requested_value is not None and requested_value < minimum:
            if allow_unsafe:
                active_logger.warning(
                    "%s below safe floor (requested=%.1fs, minimum=%.1fs) but %s=1 allows override",
                    env_var,
                    requested_value,
                    minimum,
                    _OVERRIDE_ENV,
                )
                override_granted = True
                effective_value = requested_value
            elif prompt_override is not None:
                override_granted = prompt_override(env_var, requested_value, minimum)
                if override_granted:
                    active_logger.warning(
                        "%s below safe floor (requested=%.1fs, minimum=%.1fs); proceeding after explicit user override",
                        env_var,
                        requested_value,
                        minimum,
                    )
                    effective_value = requested_value
                else:
                    clamped = True
                    effective_value = minimum
            else:
                clamped = True
                effective_value = minimum

            if clamped:
                active_logger.warning(
                    "%s below safe floor (requested=%.1fs); clamping to %.1fs",
                    env_var,
                    requested_value,
                    minimum,
                    extra={
                        "requested_timeout": requested_value,
                        "timeout_floor": minimum,
                        "effective_timeout": effective_value,
                    },
                )
                os.environ[env_var] = str(effective_value)

        results[env_var] = {
            "requested": requested_value,
            "effective": effective_value,
            "minimum": minimum,
            "clamped": clamped,
            "override_granted": override_granted,
        }

    results[_OVERRIDE_ENV] = {"requested": float(allow_unsafe), "effective": float(allow_unsafe)}
    return results


def render_prepare_pipeline_timeout_hints(vector_heavy: bool | None = None) -> list[str]:
    """Return standard remediation hints for ``prepare_pipeline_for_bootstrap`` timeouts."""

    vector_wait = _parse_float(os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS")) or _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"
    ]
    vector_step = _parse_float(os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT")) or _BOOTSTRAP_TIMEOUT_MINIMUMS[
        "BOOTSTRAP_VECTOR_STEP_TIMEOUT"
    ]

    hints = [
        "Increase MENACE_BOOTSTRAP_WAIT_SECS=240 or BOOTSTRAP_STEP_TIMEOUT=240 for slower bootstrap hosts.",
        (
            "Vector-heavy pipelines: set MENACE_BOOTSTRAP_VECTOR_WAIT_SECS="
            f"{int(vector_wait)} or BOOTSTRAP_VECTOR_STEP_TIMEOUT={int(vector_step)} "
            "to bypass the legacy 30s cap and give vector services time to warm up."
        ),
        "Stagger concurrent bootstraps or shrink watched directories to reduce contention during pipeline and vector service startup.",
    ]

    if vector_heavy:
        return list(hints)

    return hints


class SharedTimeoutCoordinator:
    """Coordinate shared timeout budgets across related bootstrap tasks.

    The coordinator exposes a lightweight API that allows modules to reserve a
    time slice from a shared budget before starting heavy work (for example
    vectorizer warm-ups, retriever hydration, database bootstrap, or
    orchestrator state loads). Reservations are serialized to prevent unrelated
    helpers from racing the same global deadline, and detailed consumption
    metadata is logged to aid debugging.
    """

    def __init__(
        self,
        total_budget: float | None,
        *,
        logger: logging.Logger | None = None,
        namespace: str = "bootstrap",
    ) -> None:
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.namespace = namespace
        self.logger = logger or LOGGER
        self._lock = threading.Lock()
        self._timeline: list[dict[str, float | str | None]] = []

    def _reserve(
        self,
        label: str,
        requested: float | None,
        minimum: float,
        metadata: Mapping[str, object] | None,
    ) -> tuple[float | None, MutableMapping[str, object]]:
        with self._lock:
            effective = requested if requested is not None else minimum
            effective = max(minimum, effective)
            remaining_before = self.remaining_budget
            if self.remaining_budget is not None:
                effective = min(effective, max(self.remaining_budget, 0.0))
                self.remaining_budget = max(self.remaining_budget - effective, 0.0)

            record: MutableMapping[str, object] = {
                "label": label,
                "requested": requested,
                "minimum": minimum,
                "effective": effective,
                "remaining_before": remaining_before,
                "remaining_after": self.remaining_budget,
                "namespace": self.namespace,
            }
            if metadata:
                record.update({f"meta.{k}": v for k, v in metadata.items()})

        self.logger.info(
            "shared timeout budget reserved",
            extra={"shared_timeout": dict(record)},
        )
        return effective, record

    @contextlib.contextmanager
    def consume(
        self,
        label: str,
        *,
        requested: float | None,
        minimum: float = 0.0,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[tuple[float | None, Mapping[str, object]]]:
        """Reserve a time slice and log consumption when complete."""

        start = time.monotonic()
        effective, record = self._reserve(label, requested, minimum, metadata)
        try:
            yield effective, record
        finally:
            elapsed = time.monotonic() - start
            record = dict(record)
            record.update({"elapsed": elapsed, "namespace": self.namespace})
            with self._lock:
                self._timeline.append(record)
            self.logger.info(
                "shared timeout budget consumed",
                extra={"shared_timeout": record},
            )

    def snapshot(self) -> Mapping[str, object]:
        """Return a shallow snapshot of coordinator state."""

        with self._lock:
            return {
                "namespace": self.namespace,
                "total_budget": self.total_budget,
                "remaining_budget": self.remaining_budget,
                "timeline": list(self._timeline),
            }
