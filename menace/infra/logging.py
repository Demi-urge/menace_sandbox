"""Deterministic structured logging helpers backed by LoggingError."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

from menace.errors.exceptions import LoggingError


def get_logger(name: str) -> dict[str, Any]:
    """Return a configured logger wrapped in a structured response.

    Args:
        name: Logger name used by the standard library logging registry.

    Returns:
        A dict with schema:
            {
                "status": "ok",
                "data": {"logger": logging.Logger},
                "errors": [],
                "metadata": {"name": str},
            }

    Raises:
        LoggingError: If ``name`` is empty or invalid.
    """

    if not isinstance(name, str) or not name:
        raise LoggingError(
            "Logger name must be a non-empty string.",
            details={"name": name},
        )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not _has_structured_handler(logger):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)

    return {
        "status": "ok",
        "data": {"logger": logger},
        "errors": [],
        "metadata": {"name": name},
    }


def log_event(
    logger: logging.Logger,
    event: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a structured log entry and return a structured response.

    Args:
        logger: Logger obtained from ``get_logger(...)[\"data\"][\"logger\"]``.
        event: Event identifier or message to emit.
        context: Optional structured context for the log entry. If ``context``
            includes a ``timestamp`` key, it must be a valid ISO-8601 string
            with timezone information and will be normalized to UTC for the
            structured payload; otherwise no timestamp field is emitted.

    Returns:
        A dict with schema:
            {
                "status": "ok",
                "data": {"event": str, "context": dict[str, Any]},
                "errors": [],
                "metadata": {"context_keys": list[str]},
            }

    Raises:
        LoggingError: If ``event`` is empty, ``logger`` is invalid, or
            ``context`` is not a mapping with string keys. Invalid timestamps
            in ``context`` also raise ``LoggingError``.
    """

    if not isinstance(logger, logging.Logger):
        raise LoggingError(
            "Logger must be an instance of logging.Logger.",
            details={"logger_type": type(logger).__name__},
        )
    if not isinstance(event, str) or not event:
        raise LoggingError(
            "Event must be a non-empty string.",
            details={"event": event},
        )

    normalized_context = _normalize_context(context)
    extra: dict[str, Any] = {"event": event, "context": normalized_context}

    logger.info(event, extra=extra)

    return {
        "status": "ok",
        "data": {"event": event, "context": dict(normalized_context)},
        "errors": [],
        "metadata": {"context_keys": list(normalized_context.keys())},
    }


class _StructuredFormatter(logging.Formatter):
    """Formatter that renders log records as deterministic JSON."""

    def format(self, record: logging.LogRecord) -> str:
        raw_context = getattr(record, "context", {})
        context = dict(raw_context) if isinstance(raw_context, Mapping) else {}
        timestamp = context.pop("timestamp", None)
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
            "context": context,
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp

        return json.dumps(
            payload,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )


def _has_structured_handler(logger: logging.Logger) -> bool:
    """Return True if the logger already has the structured handler.

    Args:
        logger (logging.Logger): Logger instance to inspect.

    Returns:
        bool: ``True`` when a ``_StructuredFormatter`` is attached.

    Raises:
        None: This helper does not raise.

    Invariants:
        - Inspection is deterministic for a given logger configuration.
        - No mutation of the logger occurs.
    """

    for handler in logger.handlers:
        formatter = handler.formatter
        if isinstance(formatter, _StructuredFormatter):
            return True
    return False


def _normalize_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize optional context mapping for structured log payloads.

    Args:
        context (Mapping[str, Any] | None): Optional mapping of context values.

    Returns:
        dict[str, Any]: Normalized context dictionary (empty if ``None``).

    Raises:
        LoggingError: If ``context`` is not a mapping or contains invalid keys.
            Invalid ``timestamp`` values also raise ``LoggingError``.

    Invariants:
        - Context keys must be non-empty strings.
        - ``timestamp`` values are normalized to UTC ISO-8601 strings.
        - No mutation of the input mapping occurs.
        - Deterministic for identical input mappings.
    """

    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise LoggingError(
            "Context must be a mapping when provided.",
            details={"context_type": type(context).__name__},
        )

    normalized: dict[str, Any] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not key:
            raise LoggingError(
                "Context keys must be non-empty strings.",
                details={"context_key": key},
            )
        if key == "timestamp":
            normalized[key] = _normalize_timestamp(value)
        else:
            normalized[key] = value
    return normalized


def _format_timestamp(created: float) -> str:
    """Format epoch seconds into a deterministic ISO-8601 timestamp.

    Args:
        created (float): Epoch seconds to format.

    Returns:
        str: ISO-8601 timestamp in UTC with timezone offset.

    Raises:
        None: This helper does not raise.

    Invariants:
        - Output is deterministic for the given ``created`` value.
        - Timestamp is always expressed in UTC.
    """

    return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()


def _normalize_timestamp(value: Any) -> str:
    """Normalize a caller-provided timestamp into UTC ISO-8601 format.

    Args:
        value (Any): Timestamp value expected to be an ISO-8601 string with tzinfo.

    Returns:
        str: Normalized ISO-8601 timestamp in UTC with timezone offset.

    Raises:
        LoggingError: If the timestamp is not a valid ISO-8601 string with tzinfo.
    """

    if not isinstance(value, str) or not value:
        raise LoggingError(
            "Timestamp must be a non-empty ISO-8601 string with timezone data.",
            details={"timestamp_type": type(value).__name__},
        )
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise LoggingError(
            "Timestamp must be a valid ISO-8601 string with timezone data.",
            details={"timestamp": value},
        ) from exc
    if parsed.tzinfo is None:
        raise LoggingError(
            "Timestamp must include timezone information.",
            details={"timestamp": value},
        )
    return parsed.astimezone(timezone.utc).isoformat()
