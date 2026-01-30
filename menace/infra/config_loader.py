"""Configuration loading utilities for infrastructure components."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from menace.errors.exceptions import ConfigError


def load_config(config: Mapping[str, Any], required_keys: Sequence[str]) -> Dict[str, Any]:
    """Validate and normalize configuration input.

    This function is pure and deterministic. It validates required keys, rejects
    ``None`` values, and returns the canonical ``{status, data, errors, meta}``
    schema with a shallow copy of the input configuration.

    Args:
        config: Raw configuration mapping.
        required_keys: Keys that must exist in ``config`` with non-``None`` values.

    Returns:
        A canonical response payload with normalized configuration data.

    Raises:
        ConfigError: If the configuration is missing required keys, contains
            ``None`` values for required keys, or is not a mapping.
    """

    if config is None:
        raise ConfigError(
            message="Config must be a mapping, not None.",
            details={"config": None},
        )

    if not isinstance(config, Mapping):
        raise ConfigError(
            message="Config must be a mapping.",
            details={"config_type": type(config).__name__},
        )

    if required_keys is None:
        raise ConfigError(
            message="required_keys must be provided.",
            details={"required_keys": None},
        )

    missing_keys = [
        key
        for key in required_keys
        if key not in config or config.get(key) is None
    ]
    if missing_keys:
        raise ConfigError(
            message="Missing required configuration keys.",
            details={"missing_keys": missing_keys},
        )

    required_key_set = set(required_keys)
    extra_keys = [key for key in config.keys() if key not in required_key_set]

    data = dict(config)
    meta = {
        "missing_keys": [],
        "extra_keys": extra_keys,
    }

    return {
        "status": "ok",
        "data": data,
        "errors": [],
        "meta": meta,
    }
