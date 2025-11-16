from __future__ import annotations

"""Central configuration loader for Security AI."""

import json
import logging
import os
from types import MappingProxyType
from typing import Any, Mapping

import yaml
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = resolve_path("config/security_ai_config.json")

# Keys required in the configuration
_REQUIRED_KEYS = {
    "log_paths": dict,
    "risk_thresholds": dict,
    "forbidden_domains": list,
    "domain_risk_map": dict,
    "reward_penalties": dict,
    "override_flags": dict,
    "stripe_enabled": bool,
}

_CONFIG: MappingProxyType | None = None
_CONFIG_PATH: str | None = None


def _create_template(path: str) -> None:
    """Create a template configuration file at *path* if missing."""
    template = {
        "log_paths": {"events": str(resolve_path("logs/events.log"))},
        "risk_thresholds": {"low": 0.2, "high": 0.8},
        "forbidden_domains": ["example.com"],
        "domain_risk_map": {"example.com": 1.0},
        "reward_penalties": {"bypass_attempt": -5},
        "override_flags": {"feature_x_enabled": False},
        "stripe_enabled": False,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2)


def _validate_config(data: dict[str, Any]) -> None:
    """Validate that *data* contains all required keys with correct types."""
    missing: list[str] = []
    for key, typ in _REQUIRED_KEYS.items():
        if key not in data:
            missing.append(key)
            continue
        if not isinstance(data[key], typ):
            raise TypeError(f"Config key '{key}' must be of type {typ.__name__}")
    if missing:
        raise KeyError(f"Config missing required keys: {', '.join(missing)}")


def load_config(config_path: str = _DEFAULT_PATH) -> MappingProxyType:
    """Load and validate configuration from ``config_path``."""
    global _CONFIG, _CONFIG_PATH
    if _CONFIG is not None and config_path == _CONFIG_PATH:
        return _CONFIG

    if not os.path.exists(config_path):
        logger.warning("Config not found at %s; creating template", config_path)
        _create_template(config_path)
        raise FileNotFoundError(
            f"Template config created at {config_path}; please customize it"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    _validate_config(data)

    # lock critical structures
    data["forbidden_domains"] = frozenset(data["forbidden_domains"])
    data["domain_risk_map"] = MappingProxyType(dict(data["domain_risk_map"]))

    _CONFIG = MappingProxyType(data)
    _CONFIG_PATH = config_path
    return _CONFIG


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Return value from loaded config using dot notation lookup."""
    cfg = _CONFIG or load_config(_CONFIG_PATH or _DEFAULT_PATH)
    current: Any = cfg
    for part in key_path.split("."):
        if isinstance(current, MappingProxyType) or isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            return default
    return current


_IMPACT_MAP: Mapping[str, float] | None = None
_IMPACT_PATH: str | None = None


def _load_impact_severity_map(
    path: str | None = None,
) -> Mapping[str, float]:
    """Return impact severity mapping from ``path`` or defaults.

    The mapping defines numeric risk weights for workflow types. It defaults to
    reading :mod:`config/impact_severity.yaml` but honours the
    ``IMPACT_SEVERITY_CONFIG`` environment variable for overrides.
    """

    global _IMPACT_MAP, _IMPACT_PATH
    defaults: dict[str, float] = {
        "experimental": 0.2,
        "standard": 0.5,
        "critical": 0.9,
    }
    cfg_path = path or os.getenv(
        "IMPACT_SEVERITY_CONFIG",
        resolve_path("config/impact_severity.yaml"),
    )
    if _IMPACT_MAP is not None and cfg_path == _IMPACT_PATH:
        return _IMPACT_MAP
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, Mapping):
            overrides = {
                str(k): float(v)
                for k, v in data.items()
                if isinstance(v, (int, float))
            }
        else:
            overrides = {}
    except FileNotFoundError as exc:
        logger.warning(
            "Impact severity config file missing",
            extra={"path": cfg_path},
            exc_info=exc,
        )
        overrides = {}
    except (yaml.YAMLError, ValueError) as exc:
        logger.warning(
            "Failed to parse impact severity config",
            extra={"path": cfg_path},
            exc_info=exc,
        )
        overrides = {}
    except Exception:
        logger.exception(
            "Unexpected error loading impact severity config",
            extra={"path": cfg_path},
        )
        raise
    defaults.update(overrides)
    _IMPACT_MAP = defaults
    _IMPACT_PATH = cfg_path
    return defaults


def get_impact_severity(workflow_type: str, path: str | None = None) -> float:
    """Return impact severity for ``workflow_type``.

    ``workflow_type`` is looked up in the loaded impact severity mapping. If
    the key is missing, the ``standard`` severity is used as a fallback.
    ``path`` may be supplied to load an alternative configuration file.
    """

    mapping = _load_impact_severity_map(path)
    return float(mapping.get(workflow_type, mapping.get("standard", 0.5)))


def impact_severity_map(path: str | None = None) -> Mapping[str, float]:
    """Expose the loaded impact severity mapping."""

    return _load_impact_severity_map(path)


__all__ = [
    "load_config",
    "get_config_value",
    "get_impact_severity",
    "impact_severity_map",
    "_DEFAULT_PATH",
]
