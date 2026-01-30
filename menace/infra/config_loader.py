"""Configuration loading utilities for infrastructure components.

Schema (required fields):
    - service_name (str): Name of the service or component.
    - environment (str): Deployment environment identifier (e.g., "dev").
    - log_level (str): Logging verbosity label (e.g., "INFO").
    - timeout_seconds (int): Timeout in seconds for infra operations.
"""

from __future__ import annotations

from typing import Any, TypedDict

from menace.errors.exceptions import ConfigError


class InfraConfig(TypedDict):
    """Typed schema for the minimum required infra configuration."""

    service_name: str
    environment: str
    log_level: str
    timeout_seconds: int


_REQUIRED_SCHEMA: dict[str, type] = {
    "service_name": str,
    "environment": str,
    "log_level": str,
    "timeout_seconds": int,
}


def _is_valid_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a config mapping and return a normalized response payload.

    Args:
        config: Configuration mapping already loaded by the caller.

    Returns:
        A canonical response payload with the following schema:

        - ``status`` (str): ``"ok"`` when validation succeeds.
        - ``data`` (dict): The validated configuration.
        - ``errors`` (list): Empty list for successful loads.
        - ``meta`` (dict): Metadata about validation and schema expectations.

    Raises:
        ConfigError: If the config mapping is missing, invalid, has ``None``
            values, or contains incorrect types for required fields.
    """

    if config is None:
        raise ConfigError(
            message="config must be provided.",
            details={"config": None},
        )

    if not isinstance(config, dict):
        raise ConfigError(
            message="config must be a dict.",
            details={"config_type": type(config).__name__},
        )

    missing_keys = [key for key in _REQUIRED_SCHEMA if key not in config]
    none_keys = [key for key in _REQUIRED_SCHEMA if key in config and config[key] is None]

    invalid_types: dict[str, dict[str, str]] = {}
    for key, expected_type in _REQUIRED_SCHEMA.items():
        if key not in config or config.get(key) is None:
            continue
        value = config[key]
        if expected_type is int:
            if not _is_valid_int(value):
                invalid_types[key] = {
                    "expected": "int",
                    "actual": type(value).__name__,
                }
        elif not isinstance(value, expected_type):
            invalid_types[key] = {
                "expected": expected_type.__name__,
                "actual": type(value).__name__,
            }

    if missing_keys or none_keys or invalid_types:
        raise ConfigError(
            message="Invalid configuration payload.",
            details={
                "missing_keys": missing_keys,
                "none_keys": none_keys,
                "invalid_types": dict(sorted(invalid_types.items())),
            },
        )

    required_data = {key: config[key] for key in _REQUIRED_SCHEMA}
    extra_keys = sorted(key for key in config.keys() if key not in _REQUIRED_SCHEMA)
    normalized_data = {**required_data, **{key: config[key] for key in extra_keys}}

    meta = {
        "required_keys": list(_REQUIRED_SCHEMA.keys()),
        "validated_keys": sorted(_REQUIRED_SCHEMA.keys()),
        "extra_keys": extra_keys,
    }

    return {
        "status": "ok",
        "data": normalized_data,
        "errors": [],
        "meta": meta,
    }
