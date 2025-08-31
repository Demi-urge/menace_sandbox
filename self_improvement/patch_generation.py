from __future__ import annotations

"""Patch generation utilities for the self-improvement engine."""

import logging

from .utils import _load_callable, _call_with_retries

try:  # pragma: no cover - simplified environments
    from ..logging_utils import log_record
except Exception:  # pragma: no cover - best effort
    def log_record(**fields: object) -> dict[str, object]:  # type: ignore
        return fields


def generate_patch(
    *args: object, retries: int = 3, delay: float = 0.1, **kwargs: object
) -> int:
    """Generate a patch via :mod:`quick_fix_engine`.

    This wrapper ensures the optional dependency is available and converts
    missing or unsuccessful patch generation into a structured
    :class:`RuntimeError` with logging instead of returning ``None``.
    """

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
            "quick_fix_engine is required for patch generation. Install it via `pip install quick_fix_engine`."
        ) from exc
    try:
        patch_id = _call_with_retries(func, *args, retries=retries, delay=delay, **kwargs)
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
