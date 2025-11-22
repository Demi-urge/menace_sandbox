"""Compatibility shim for sandbox logging utilities.

This module bridges imports inside :mod:`sandbox_runner` to the shared
top-level :mod:`logging_utils` module. Packaging the shim avoids
``ModuleNotFoundError`` when the sandbox is installed without the project
root on ``sys.path`` (for example when running ``python -m
sandbox_runner.cli``).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict


def _safe_log_record(**fields: Any) -> Dict[str, Any]:
    """Minimal ``log_record`` fallback used when the core helper is absent."""

    return {key: value for key, value in fields.items() if value is not None}


try:
    _core = importlib.import_module("logging_utils")
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight installs
    def get_logger(name: str) -> logging.Logger:
        """Return a basic logger when shared logging is unavailable."""

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

    def setup_logging(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401
        """No-op placeholder when shared setup is missing."""

        return None

    def set_correlation_id(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401
        """No-op placeholder when shared correlation handling is missing."""

        return None

    log_record = _safe_log_record
else:  # pragma: no cover - delegates to shared implementation
    get_logger = _core.get_logger
    setup_logging = getattr(_core, "setup_logging", lambda *_a, **_k: None)
    set_correlation_id = getattr(_core, "set_correlation_id", lambda *_a, **_k: None)
    log_record = getattr(_core, "log_record", _safe_log_record)


__all__ = [
    "get_logger",
    "log_record",
    "set_correlation_id",
    "setup_logging",
]
