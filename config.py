"""Application configuration loader.

Loads base settings from ``config/settings.yaml`` and overlays profile-specific
values from ``config/<mode>.yaml``. The active mode is resolved from the
``--mode`` command-line argument or the ``IGI_MODE`` environment variable (which
defaults to ``dev``).

The configuration schema is validated using Pydantic models to provide helpful
error messages for missing or invalid fields.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Pydantic models defining the configuration schema
# ---------------------------------------------------------------------------


class Paths(BaseModel):
    """File system locations used by the application."""

    data_dir: str
    log_dir: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("data_dir", "log_dir")
    @classmethod
    def _not_empty(cls, value: str, info) -> str:  # type: ignore[override]
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class Thresholds(BaseModel):
    """Operational thresholds expressed as floats between 0 and 1."""

    error: float = Field(ge=0.0, le=1.0)
    alert: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_order(self) -> "Thresholds":
        if self.alert <= self.error:
            raise ValueError("alert must be greater than error")
        return self


class APIKeys(BaseModel):
    """External service authentication keys."""

    openai: str
    serp: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("openai", "serp")
    @classmethod
    def _non_blank(cls, value: str, info) -> str:  # type: ignore[override]
        if not value or "REPLACE" in value:
            raise ValueError(f"{info.field_name} API key must be provided")
        return value


class Logging(BaseModel):
    """Logging configuration."""

    verbosity: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    model_config = ConfigDict(extra="forbid")


class Vector(BaseModel):
    """Vector search parameters."""

    dimensions: int = Field(gt=0)
    distance_metric: str

    model_config = ConfigDict(extra="forbid")


class Bot(BaseModel):
    """Bot tuning parameters."""

    learning_rate: float = Field(gt=0)
    epsilon: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    """Top-level application configuration."""

    paths: Paths
    thresholds: Thresholds
    api_keys: APIKeys
    logging: Logging
    vector: Vector
    bot: Bot

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # Runtime modification helpers
    # ------------------------------------------------------------------

    def apply_overrides(self, mapping: Dict[str, Any]) -> "Config":
        """Return a new ``Config`` with *mapping* merged into the current data."""

        data = self.model_dump()
        _merge_dict(data, mapping)
        return Config.model_validate(data)


# ---------------------------------------------------------------------------
# Configuration loading utilities
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_SETTINGS_FILE = CONFIG_DIR / "settings.yaml"

_MODE: str | None = None
_CONFIG_PATH: Path | None = None
_OVERRIDES: Dict[str, Any] = {}
CONFIG: Config | None = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*."""
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    mode: str | None = None,
    config_file: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Config:
    """Load the configuration for the given *mode*.

    Parameters
    ----------
    mode:
        Optional profile name such as ``"dev"`` or ``"prod"``. When ``None`` the
        value is read from the ``IGI_MODE`` environment variable or falls back to
        ``"dev"``.
    config_file:
        Optional path to an additional configuration file that will be merged
        after the profile-specific settings.
    overrides:
        Mapping of configuration values to override using dotted keys.
    """

    active_mode = mode or os.getenv("IGI_MODE", "dev")

    data = _load_yaml(DEFAULT_SETTINGS_FILE)

    profile_file = CONFIG_DIR / f"{active_mode}.yaml"
    if profile_file.exists():
        data = _merge_dict(data, _load_yaml(profile_file))
    else:
        raise FileNotFoundError(
            f"Config profile '{active_mode}' not found at {profile_file}"
        )

    if config_file:
        data = _merge_dict(data, _load_yaml(Path(config_file)))

    cfg = Config.model_validate(data)
    if overrides:
        cfg = cfg.apply_overrides(overrides)
    return cfg


# Initial global configuration instance
CONFIG = load_config()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _build_overrides(pairs: list[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}', expected key=value")
        key, value = pair.split("=", 1)
        value_data = yaml.safe_load(value)
        current = result
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value_data
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load configuration")
    parser.add_argument("--mode", help="Configuration mode (e.g. dev or prod)")
    parser.add_argument(
        "--config", help="Additional configuration YAML file to merge", dest="config_file"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in key=value format",
    )
    return parser.parse_args(argv)


def reload() -> Config:
    """Reload configuration from disk using stored parameters."""

    global CONFIG
    CONFIG = load_config(_MODE, _CONFIG_PATH, _OVERRIDES)
    return CONFIG


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    global _MODE, _CONFIG_PATH, _OVERRIDES
    _MODE = args.mode
    _CONFIG_PATH = Path(args.config_file) if args.config_file else None
    _OVERRIDES = _build_overrides(args.overrides or [])
    cfg = reload()
    print(cfg.model_dump())


if __name__ == "__main__":  # pragma: no cover
    main()
