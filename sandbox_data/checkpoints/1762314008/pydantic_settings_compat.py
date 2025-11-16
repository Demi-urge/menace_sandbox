"""Compatibility helpers for optional pydantic-settings dependency."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - prefer real dependency when available
    from pydantic_settings import BaseSettings as _BaseSettings  # type: ignore
    from pydantic_settings import SettingsConfigDict as _SettingsConfigDict  # type: ignore
except Exception:  # pragma: no cover - fall back when package missing or incompatible
    _BaseSettings = None  # type: ignore[assignment]
    _SettingsConfigDict = None  # type: ignore[assignment]
else:
    BaseSettings = _BaseSettings  # type: ignore[misc]
    SettingsConfigDict = _SettingsConfigDict  # type: ignore[misc]
    PYDANTIC_V2 = True

if _BaseSettings is None:
    from pydantic import BaseModel

    try:  # pragma: no cover - available on pydantic>=2
        from pydantic import ConfigDict
    except Exception:  # pragma: no cover - pydantic<2 compatibility
        ConfigDict = None  # type: ignore[assignment]

    def _normalise_aliases(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            result: list[str] = []
            for item in value:
                result.extend(_normalise_aliases(item))
            return result
        try:
            iterator = iter(value)  # type: ignore[arg-type]
        except TypeError:
            return [str(value)]
        else:
            result = []
            for item in iterator:
                result.extend(_normalise_aliases(item))
            return result

    def _read_env_files(candidate: Any) -> dict[str, str]:
        paths: list[Path] = []
        if not candidate:
            return {}
        if isinstance(candidate, (str, os.PathLike)):
            paths.append(Path(candidate).expanduser())
        elif isinstance(candidate, Iterable):
            for entry in candidate:
                if isinstance(entry, (str, os.PathLike)):
                    paths.append(Path(entry).expanduser())
        values: dict[str, str] = {}
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.lower().startswith("export "):
                            line = line[7:].lstrip()
                        key, sep, value = line.partition("=")
                        if not sep:
                            continue
                        key = key.strip()
                        if not key:
                            continue
                        value = value.strip()
                        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                            value = value[1:-1]
                        values.setdefault(key, value)
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return values

    class _FallbackBaseSettings(BaseModel):
        """Minimal replacement for :class:`pydantic_settings.BaseSettings`."""

        if ConfigDict is not None:  # pragma: no cover - pydantic>=2
            model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)  # type: ignore[call-arg]
        else:  # pragma: no cover - pydantic<2 fallback
            model_config: Mapping[str, Any] | None = {"extra": "ignore"}

        def __init__(self, **data: Any) -> None:  # pragma: no cover - configuration dependent
            values = self._build_settings_values()
            values.update(data)
            super().__init__(**values)

        @classmethod
        def _collect_field_env(cls) -> dict[str, list[str]]:
            field_env: dict[str, list[str]] = {}
            for name, field in cls.model_fields.items():  # type: ignore[attr-defined]
                keys: list[str] = []
                keys.extend(_normalise_aliases(getattr(field, "validation_alias", None)))
                keys.extend(_normalise_aliases(getattr(field, "alias", None)))
                extra = getattr(field, "json_schema_extra", None)
                if isinstance(extra, Mapping):
                    keys.extend(_normalise_aliases(extra.get("env")))
                ordered: list[str] = []
                for key in keys:
                    if key and key not in ordered:
                        ordered.append(key)
                if ordered:
                    field_env[name] = ordered
            return field_env

        @classmethod
        def _build_settings_values(cls) -> dict[str, Any]:
            values: dict[str, Any] = {}
            field_env = cls._collect_field_env()
            if not field_env:
                return values

            config = getattr(cls, "model_config", None)
            env_file = None
            if isinstance(config, Mapping):
                env_file = config.get("env_file")
            file_values = _read_env_files(env_file)

            for field_name, keys in field_env.items():
                for key in keys:
                    if key in os.environ:
                        values[field_name] = os.environ[key]
                        break
                    if key in file_values:
                        values[field_name] = file_values[key]
                        break
            return values

    BaseSettings = _FallbackBaseSettings  # type: ignore[misc]

    def SettingsConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[misc]
        return dict(**kwargs)

    try:  # pragma: no cover - detect runtime pydantic major version
        from pydantic.version import VERSION
    except Exception:  # pragma: no cover - fallback when metadata missing
        PYDANTIC_V2 = True
    else:
        PYDANTIC_V2 = VERSION.split(".")[0] >= "2"

__all__ = ["BaseSettings", "SettingsConfigDict", "PYDANTIC_V2"]
