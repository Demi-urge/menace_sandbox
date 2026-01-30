"""Deterministic structured logging helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LogEvent:
    """Structured log event payload.

    Attributes:
        level: Log level name.
        message: Human-readable message.
        metadata: Structured metadata to attach to the event.
    """

    level: str
    message: str
    metadata: dict[str, Any]


class LogEventError(ValueError):
    """Base class for log event validation errors."""


class InvalidLogLevelError(LogEventError):
    """Raised when a log level is not accepted."""


class InvalidLogEventError(LogEventError):
    """Raised when a log event payload is invalid."""


_ALLOWED_LEVELS: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error", "critical"}
)


def log_event(level: str, message: str, metadata: Mapping[str, Any] | None) -> None:
    """Emit a structured log event to stdout.

    Invariants:
        * The function is deterministic for a given input.
        * Accepted levels are: debug, info, warning, error, critical.
        * No global mutable state is read or mutated.

    Args:
        level: The severity level for the event.
        message: Human-readable event message.
        metadata: Structured metadata to include in the payload.

    Raises:
        InvalidLogLevelError: If ``level`` is not accepted.
        InvalidLogEventError: If the log payload cannot be serialized.
    """

    if level not in _ALLOWED_LEVELS:
        raise InvalidLogLevelError(
            f"Invalid log level '{level}'. Allowed: {sorted(_ALLOWED_LEVELS)}"
        )
    if not isinstance(message, str) or not message:
        raise InvalidLogEventError("message must be a non-empty string")

    metadata_dict = _normalize_metadata(metadata)
    event = LogEvent(level=level, message=message, metadata=metadata_dict)

    payload = {
        "level": event.level,
        "message": event.message,
        "metadata": event.metadata,
    }

    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidLogEventError("metadata must be JSON-serializable") from exc

    sys.stdout.write(serialized + "\n")
    sys.stdout.flush()


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize metadata mapping to a dictionary."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise InvalidLogEventError("metadata must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise InvalidLogEventError("metadata keys must be non-empty strings")
        normalized[key] = value
    return normalized
