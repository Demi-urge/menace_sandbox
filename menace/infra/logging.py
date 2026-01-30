"""Deterministic structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Mapping


class StructuredLogError(ValueError):
    """Raised when structured logging inputs are invalid."""


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for deterministic structured output.

    Invariants:
        * The logger uses a formatter that emits deterministic JSON.
        * No timestamps are included unless explicitly provided in event data.
        * Repeated calls with the same name do not add duplicate handlers.

    Args:
        name: Logger name used by the standard library logging registry.

    Returns:
        A configured :class:`logging.Logger` instance.
    """

    logger = logging.getLogger(name)
    logger.propagate = False

    if not _has_structured_handler(logger):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)

    return logger


def log_event(
    logger: logging.Logger,
    level: int | str,
    event: str,
    status: str | None = None,
    data: Mapping[str, Any] | None = None,
    errors: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured event using the provided logger.

    Invariants:
        * ``event`` is a non-empty string.
        * ``data``, ``errors``, and ``metadata`` are mappings when provided.
        * Output is deterministic JSON with sorted keys.
        * No global mutable state is accessed or mutated outside logger config.

    Args:
        logger: Logger obtained from :func:`get_logger`.
        level: Logging level name or numeric value.
        event: Event name for the log entry.
        status: Optional status descriptor (e.g., success, failure).
        data: Optional structured payload data.
        errors: Optional structured error details.
        metadata: Optional metadata values for the event.

    Raises:
        StructuredLogError: If required fields are missing or invalid.
    """

    if not isinstance(event, str) or not event:
        raise StructuredLogError("event must be a non-empty string")

    payload: dict[str, Any] = {
        "event": event,
        "status": status,
        "data": _normalize_mapping("data", data),
        "errors": _normalize_mapping("errors", errors),
        "metadata": _normalize_mapping("metadata", metadata),
    }

    logger.log(_resolve_level(level), "", extra=payload)


class _StructuredFormatter(logging.Formatter):
    """Formatter that renders log records as deterministic JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "logger": record.name,
            "level": record.levelname.lower(),
        }

        for key in ("event", "status", "data", "errors", "metadata"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        message = record.getMessage()
        if message:
            payload["message"] = message

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


def _normalize_mapping(name: str, mapping: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize optional mappings for structured log payloads."""

    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise StructuredLogError(f"{name} must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise StructuredLogError(f"{name} keys must be non-empty strings")
        normalized[key] = value
    return normalized


def _resolve_level(level: int | str) -> int:
    """Resolve a logging level from an int or case-insensitive name."""

    if isinstance(level, int):
        return level
    if not isinstance(level, str) or not level:
        raise StructuredLogError("level must be a logging level name or integer")

    resolved = logging.getLevelName(level.upper())
    if isinstance(resolved, int):
        return resolved

    raise StructuredLogError(f"Unknown logging level: {level}")
