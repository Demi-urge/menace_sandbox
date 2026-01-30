"""Deterministic structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


class StructuredLogError(ValueError):
    """Raised when structured logging inputs are invalid."""


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for deterministic structured output.

    Args:
        name: Logger name used by the standard library logging registry.

    Returns:
        A configured :class:`logging.Logger` instance.

    Raises:
        StructuredLogError: If ``name`` is empty.
    """

    if not isinstance(name, str) or not name:
        raise StructuredLogError("name must be a non-empty string")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not _has_structured_handler(logger):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)

    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured log entry using the provided logger.

    Args:
        logger: Logger obtained from :func:`get_logger`.
        event: Event identifier or message to emit.
        context: Optional structured context for the log entry.

    Raises:
        StructuredLogError: If ``event`` is empty, ``logger`` is invalid, or
            ``context`` is not a mapping with string keys.
    """

    if not isinstance(logger, logging.Logger):
        raise StructuredLogError("logger must be a logging.Logger")
    if not isinstance(event, str) or not event:
        raise StructuredLogError("event must be a non-empty string")

    normalized_context = _normalize_context(context)
    extra: dict[str, Any] = {"event": event, "context": normalized_context}

    logger.info(event, extra=extra)


class _StructuredFormatter(logging.Formatter):
    """Formatter that renders log records as deterministic JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _format_timestamp(record.created),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
            "context": getattr(record, "context", {}),
        }

        return json.dumps(
            payload,
            sort_keys=False,
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


def _normalize_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize optional context mapping for structured log payloads."""

    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise StructuredLogError("context must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not key:
            raise StructuredLogError("context keys must be non-empty strings")
        normalized[key] = value
    return normalized


def _format_timestamp(created: float) -> str:
    """Format epoch seconds into a deterministic ISO-8601 timestamp."""

    return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
