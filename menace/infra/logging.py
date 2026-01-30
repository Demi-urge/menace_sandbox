"""Deterministic structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Mapping


class StructuredLogError(ValueError):
    """Raised when structured logging inputs are invalid."""


def get_logger(name: str, level: int) -> logging.Logger:
    """Return a configured logger for deterministic structured output.

    Args:
        name: Logger name used by the standard library logging registry.
        level: Numeric logging level for the logger instance.

    Returns:
        A configured :class:`logging.Logger` instance.

    Raises:
        StructuredLogError: If ``name`` is empty or ``level`` is not an integer.
    """

    if not isinstance(name, str) or not name:
        raise StructuredLogError("name must be a non-empty string")
    if not isinstance(level, int):
        raise StructuredLogError("level must be an integer logging level")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not _has_structured_handler(logger):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)

    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured log entry using the provided logger.

    Args:
        logger: Logger obtained from :func:`get_logger`.
        level: Numeric logging level for this log entry.
        message: Human-readable log message.
        meta: Optional structured metadata for the log entry.

    Raises:
        StructuredLogError: If ``message`` is empty, ``level`` is invalid, or ``meta``
            is not a mapping with string keys.
    """

    if not isinstance(level, int):
        raise StructuredLogError("level must be an integer logging level")
    if not isinstance(message, str) or not message:
        raise StructuredLogError("message must be a non-empty string")

    normalized_meta = _normalize_meta(meta)
    extra: dict[str, Any] = {}
    if normalized_meta is not None:
        extra["meta"] = normalized_meta

    logger.log(level, message, extra=extra)


class _StructuredFormatter(logging.Formatter):
    """Formatter that renders log records as deterministic JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }

        if record.name:
            payload["logger"] = record.name

        meta = getattr(record, "meta", None)
        if meta is not None:
            payload["meta"] = meta

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )


def _has_structured_handler(logger: logging.Logger) -> bool:
    """Return True if the logger already has the structured handler."""

    for handler in logger.handlers:
        formatter = handler.formatter
        if isinstance(formatter, _StructuredFormatter):
            return True
    return False


def _normalize_meta(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize optional meta mapping for structured log payloads."""

    if meta is None:
        return None
    if not isinstance(meta, Mapping):
        raise StructuredLogError("meta must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in meta.items():
        if not isinstance(key, str) or not key:
            raise StructuredLogError("meta keys must be non-empty strings")
        normalized[key] = value
    return normalized
