from __future__ import annotations

"""Patch generation utilities for the self-improvement engine.

This module provides a thin wrapper around :mod:`quick_fix_engine`'s patch
generation.  Callers must supply a ready-to-use
``vector_service.ContextBuilder`` instance when invoking
``generate_patch``.
"""

import logging

from .utils import _load_callable, _call_with_retries
from menace_sandbox.sandbox_settings import SandboxSettings
from menace_sandbox.metrics_exporter import self_improvement_failure_total
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .target_region import TargetRegion
    from vector_service.context_builder import ContextBuilder
    from menace_sandbox.self_coding_manager import SelfCodingManager

try:  # pragma: no cover - simplified environments
    from menace_sandbox.logging_utils import log_record
except (ImportError, AttributeError) as exc:  # pragma: no cover - best effort
    logging.getLogger(__name__).warning(
        "log_record unavailable", extra={"component": __name__}, exc_info=exc
    )
    self_improvement_failure_total.labels(reason="log_record_import").inc()

    def log_record(**fields: object) -> dict[str, object]:  # type: ignore
        return fields


_settings = SandboxSettings()


def generate_patch(
    module: str,
    manager: "SelfCodingManager",
    *args: object,
    context_builder: "ContextBuilder",
    target_region: "TargetRegion" | None = None,
    retries: int = _settings.patch_retries,
    delay: float = _settings.patch_retry_delay,
    **kwargs: object,
) -> int:
    """Generate a patch via :mod:`quick_fix_engine`.

    This wrapper ensures the optional dependency is available and converts
    missing or unsuccessful patch generation into a structured
    :class:`RuntimeError` with logging instead of returning ``None``.
    """

    if context_builder is None:
        raise TypeError("context_builder is required")
    logger = logging.getLogger(__name__)
    try:
        func = _load_callable("quick_fix_engine", "generate_patch")
    except RuntimeError as exc:  # pragma: no cover - best effort logging
        logger.error(
            "quick_fix_engine missing",
            extra=log_record(module=__name__),
            exc_info=exc,
        )
        raise RuntimeError(
            "quick_fix_engine is required for patch generation. "
            "Install it via `pip install quick_fix_engine`."
        ) from exc
    try:
        patch_id = _call_with_retries(
            func,
            module,
            manager,
            *args,
            context_builder=context_builder,
            retries=retries,
            delay=delay,
            target_region=target_region,
            **kwargs,
        )
    except (RuntimeError, OSError) as exc:  # pragma: no cover - best effort logging
        logger.error(
            "quick_fix_engine failed",
            extra=log_record(module=__name__),
            exc_info=exc,
        )
        raise RuntimeError("quick_fix_engine failed to generate patch") from exc
    if patch_id is None:
        logger.error("quick_fix_engine returned no patch")
        raise RuntimeError("quick_fix_engine did not produce a patch")
    return int(patch_id)


__all__ = ["generate_patch"]
