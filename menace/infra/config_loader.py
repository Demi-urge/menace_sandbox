"""Configuration loading utilities for infrastructure components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConfigSchema:
    """Normalized configuration schema.

    Attributes:
        service_name: Human-readable service identifier.
        environment: Execution environment name (e.g., "prod").
        log_level: Logging level for structured logging output.
        debug: Whether to enable debug behaviors.
        metadata: Extra static metadata to attach to log events.
    """

    service_name: str
    environment: str
    log_level: str
    debug: bool
    metadata: dict[str, str]


class ConfigError(ValueError):
    """Base class for configuration errors."""


class MissingConfigKeyError(ConfigError):
    """Raised when a required configuration key is missing."""


class InvalidConfigValueError(ConfigError):
    """Raised when a configuration value is invalid."""


_REQUIRED_KEYS: tuple[str, ...] = ("service_name", "environment", "log_level")
_ALLOWED_LOG_LEVELS: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error", "critical"}
)


def load_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize configuration input.

    This function is deterministic and performs no I/O. It validates required
    keys, rejects ``None`` values, and returns a fixed schema with defaults for
    optional settings.

    Args:
        config: Raw configuration mapping.

    Returns:
        A dictionary containing the normalized configuration schema.

    Raises:
        MissingConfigKeyError: If a required key is missing.
        InvalidConfigValueError: If a value is invalid or ``None``.
    """

    if config is None:
        raise InvalidConfigValueError("config cannot be None")

    for key in _REQUIRED_KEYS:
        if key not in config:
            raise MissingConfigKeyError(f"Missing required config key: {key}")

    for key, value in config.items():
        if value is None:
            raise InvalidConfigValueError(f"Config value for '{key}' cannot be None")

    service_name = _require_str(config, "service_name")
    environment = _require_str(config, "environment")
    log_level = _require_str(config, "log_level")
    if log_level not in _ALLOWED_LOG_LEVELS:
        raise InvalidConfigValueError(
            f"Invalid log_level '{log_level}'. Allowed: {sorted(_ALLOWED_LOG_LEVELS)}"
        )

    debug = _get_bool(config, "debug", default=False)
    metadata = _get_metadata(config, "metadata")

    normalized = ConfigSchema(
        service_name=service_name,
        environment=environment,
        log_level=log_level,
        debug=debug,
        metadata=metadata,
    )
    return {
        "service_name": normalized.service_name,
        "environment": normalized.environment,
        "log_level": normalized.log_level,
        "debug": normalized.debug,
        "metadata": dict(normalized.metadata),
    }


def _require_str(config: Mapping[str, Any], key: str) -> str:
    """Return a required string value from config."""

    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidConfigValueError(
            f"Config value for '{key}' must be a non-empty string"
        )
    return value


def _get_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    """Return an optional boolean value from config."""

    if key not in config:
        return default
    value = config.get(key)
    if not isinstance(value, bool):
        raise InvalidConfigValueError(f"Config value for '{key}' must be a boolean")
    return value


def _get_metadata(config: Mapping[str, Any], key: str) -> dict[str, str]:
    """Return metadata dict with string keys and values."""

    value = config.get(key, {})
    if not isinstance(value, dict):
        raise InvalidConfigValueError(f"Config value for '{key}' must be a dict")

    normalized: dict[str, str] = {}
    for meta_key, meta_value in value.items():
        if not isinstance(meta_key, str) or not meta_key:
            raise InvalidConfigValueError(
                f"Metadata key '{meta_key}' must be a non-empty string"
            )
        if not isinstance(meta_value, str):
            raise InvalidConfigValueError(
                f"Metadata value for '{meta_key}' must be a string"
            )
        normalized[meta_key] = meta_value
    return normalized
