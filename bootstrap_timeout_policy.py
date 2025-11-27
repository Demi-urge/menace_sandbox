"""Shared timeout policy helpers for bootstrap entry points."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict

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
