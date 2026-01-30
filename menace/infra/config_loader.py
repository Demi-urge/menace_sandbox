"""Configuration loading utilities for infrastructure components."""

from __future__ import annotations

import json
from typing import Any

from menace.errors.exceptions import ConfigError


def load_config(
    config_path: str | None,
    defaults: dict[str, Any],
    required_keys: list[str],
) -> dict[str, Any]:
    """Load configuration from a JSON file and merge with defaults.

    Args:
        config_path: Optional path to a local JSON configuration file. When
            ``None``, only ``defaults`` are used.
        defaults: Baseline configuration values. This mapping is not mutated.
        required_keys: Keys that must exist in the final configuration with
            non-``None`` values.

    Returns:
        A canonical response payload with the following schema:

        - ``status`` (str): ``"ok"`` when validation succeeds.
        - ``data`` (dict): The merged configuration.
        - ``errors`` (list): Empty list for successful loads.
        - ``meta`` (dict): Metadata about the merge (source, overrides, and
          defaults applied).

    Raises:
        ConfigError: If inputs are invalid, the JSON file cannot be read or
            parsed, the JSON payload is not a mapping, or required keys are
            missing or ``None``.
    """

    if config_path is not None and not isinstance(config_path, str):
        raise ConfigError(
            message="config_path must be a string or None.",
            details={"config_path_type": type(config_path).__name__},
        )

    if config_path is not None and not config_path.strip():
        raise ConfigError(
            message="config_path cannot be empty when provided.",
            details={"config_path": config_path},
        )

    if defaults is None:
        raise ConfigError(
            message="defaults must be provided.",
            details={"defaults": None},
        )

    if not isinstance(defaults, dict):
        raise ConfigError(
            message="defaults must be a dict.",
            details={"defaults_type": type(defaults).__name__},
        )

    if required_keys is None:
        raise ConfigError(
            message="required_keys must be provided.",
            details={"required_keys": None},
        )

    if not isinstance(required_keys, list):
        raise ConfigError(
            message="required_keys must be a list of strings.",
            details={"required_keys_type": type(required_keys).__name__},
        )

    invalid_required_keys = [
        key for key in required_keys if not isinstance(key, str) or not key
    ]
    if invalid_required_keys:
        raise ConfigError(
            message="required_keys must contain non-empty strings.",
            details={"invalid_required_keys": invalid_required_keys},
        )

    source = None
    loaded_config: dict[str, Any] = {}

    if config_path is not None:
        source = config_path
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded_config = json.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(
                message="Config file does not exist.",
                details={"config_path": config_path},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(
                message="Config file contains invalid JSON.",
                details={"config_path": config_path, "error": str(exc)},
            ) from exc
        except OSError as exc:
            raise ConfigError(
                message="Config file could not be read.",
                details={"config_path": config_path, "error": str(exc)},
            ) from exc

        if not isinstance(loaded_config, dict):
            raise ConfigError(
                message="Config file must contain a JSON object.",
                details={"config_path": config_path, "payload_type": type(loaded_config).__name__},
            )

    merged_config = {**defaults, **loaded_config}

    missing_keys = [key for key in required_keys if key not in merged_config]
    none_keys = [key for key in required_keys if merged_config.get(key) is None]
    if missing_keys or none_keys:
        raise ConfigError(
            message="Missing required configuration keys.",
            details={"missing_keys": missing_keys, "none_keys": none_keys},
        )

    overrides = sorted(loaded_config.keys())
    used_defaults = sorted(key for key in defaults.keys() if key not in loaded_config)

    meta = {
        "source": source,
        "overrides": overrides,
        "used_defaults": used_defaults,
        "missing_keys": [],
        "none_keys": [],
    }

    return {
        "status": "ok",
        "data": dict(merged_config),
        "errors": [],
        "meta": meta,
    }
