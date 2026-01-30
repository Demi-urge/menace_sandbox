"""Fallback YAML module that proxies to yaml_fallback when PyYAML is absent."""

from __future__ import annotations

from yaml_fallback import get_yaml

_yaml = get_yaml("yaml", warn=False, force_fallback=True)

YAMLError = getattr(_yaml, "YAMLError", Exception)

safe_load = _yaml.safe_load  # type: ignore[attr-defined]
load = getattr(_yaml, "load", safe_load)

safe_dump = _yaml.safe_dump  # type: ignore[attr-defined]
dump = getattr(_yaml, "dump", safe_dump)

class SafeDumper:  # pragma: no cover - compatibility stub
    pass


__all__ = ["safe_load", "safe_dump", "load", "dump", "YAMLError", "SafeDumper"]
