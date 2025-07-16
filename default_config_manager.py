from __future__ import annotations

"""Automatic environment configuration loader."""

import os
from pathlib import Path

from . import config_discovery as cd


class DefaultConfigManager:
    """Populate essential configuration values before services start."""

    def __init__(self, env_file: str = ".env") -> None:
        self.env_path = Path(env_file)

    def _load_env_file(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if not self.env_path.exists():
            return env
        for line in self.env_path.read_text().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
        return env

    def apply_defaults(self) -> None:
        """Ensure required options exist and persist them to ``.env``."""
        data = self._load_env_file()
        changed = False

        # persist already set environment values
        for name in cd._DEFAULT_VARS:
            if name in os.environ and name not in data:
                data[name] = os.environ[name]
                changed = True

        for name in cd._DEFAULT_VARS:
            val = os.environ.get(name) or data.get(name)
            if not val:
                val = cd._generate_value(name)
                data[name] = val
                changed = True
            os.environ[name] = val

        # export all values from the file to the environment
        for k, v in data.items():
            os.environ.setdefault(k, v)

        if changed or not self.env_path.exists():
            lines = [f"{k}={v}" for k, v in sorted(data.items())]
            self.env_path.write_text("\n".join(lines))

        os.environ.setdefault("MENACE_ENV_FILE", str(self.env_path))


__all__ = ["DefaultConfigManager"]
