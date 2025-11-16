"""Environment bootstrap helpers for seeding configuration defaults.

The script ``scripts/bootstrap_env.py`` is frequently executed on fresh
workstations where critical configuration values have not yet been provisioned.
This module takes care of populating sensible defaults in a cross-platform way
so that the bootstrap process can progress without spurious warnings.

Highlights
==========
* Generated secrets rely on :mod:`secrets` and satisfy basic complexity rules so
  the resulting credentials are safe to reuse during local development.
* Paths are handled through :class:`pathlib.Path` ensuring the same code works on
  Windows and POSIX machines.
* Values are persisted to the configured ``.env`` file which means subsequent
  runs reuse the same credentials instead of generating new ones every time.
"""

from __future__ import annotations

import getpass
import logging
import os
import platform
import re
import secrets
import string
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping

LOGGER = logging.getLogger(__name__)

_DEFAULT_ENV_FILENAMES = (".env", ".env.local", ".env.bootstrap")
_SENSITIVE_KEYS = {"OPENAI_API_KEY", "MENACE_PASSWORD", "POSTGRES_PASSWORD"}


class EnvironmentDefaultsManager:
    """Populate missing environment variables with deterministic defaults."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        env_file: str | os.PathLike[str] | None = None,
        environ: MutableMapping[str, str] | None = None,
    ) -> None:
        self._repo_root = repo_root or Path(__file__).resolve().parent
        self._environ = environ if environ is not None else os.environ
        self._env_path = self._resolve_env_path(env_file)
        self._file_values = self._load_env_file()
        self._environ.setdefault("MENACE_ENV_FILE", str(self._env_path))

    # ------------------------------------------------------------------
    @property
    def env_path(self) -> Path:
        return self._env_path

    # ------------------------------------------------------------------
    def ensure(self, required: Iterable[str]) -> set[str]:
        """Ensure each name in ``required`` has a value.

        The method returns the subset of variables that were generated during the
        call. The resulting set is useful for logging without exposing the
        concrete secret values.
        """

        created: dict[str, str] = {}
        for name in required:
            if self._is_defined(name):
                continue
            value = self._generate_value(name)
            self._file_values[name] = value
            self._environ[name] = value
            created[name] = value

        if created:
            self._append_to_env_file(created)
            redacted = ", ".join(sorted(self._redact(key) for key in created))
            LOGGER.info("Seeded defaults for %s", redacted)
        else:
            for name in required:
                if name in self._file_values:
                    self._environ.setdefault(name, self._file_values[name])
        return set(created)

    # ------------------------------------------------------------------
    def _resolve_env_path(
        self, env_file: str | os.PathLike[str] | None
    ) -> Path:
        if env_file:
            return Path(env_file).expanduser().resolve()
        explicit = self._environ.get("MENACE_ENV_FILE")
        if explicit:
            return Path(explicit).expanduser().resolve()
        for candidate in _DEFAULT_ENV_FILENAMES:
            path = (self._repo_root / candidate).expanduser()
            if path.exists():
                return path.resolve()
        return (self._repo_root / ".env.bootstrap").resolve()

    # ------------------------------------------------------------------
    def _load_env_file(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if not self._env_path.exists():
            return values
        try:
            for line in self._env_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                values[key.strip()] = value.strip().strip('"')
        except OSError as exc:  # pragma: no cover - diagnostics only
            LOGGER.warning("Failed to read %s: %s", self._env_path, exc)
        return values

    # ------------------------------------------------------------------
    def _append_to_env_file(self, entries: Mapping[str, str]) -> None:
        if not entries:
            return
        self._env_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if self._env_path.exists():
            try:
                existing = self._env_path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - best effort
                existing = ""
        line_sep = os.linesep or "\n"
        prefix = ""
        if existing and not existing.endswith(("\n", "\r")):
            prefix = line_sep
        text = line_sep.join(
            f"{key}={self._escape(value)}" for key, value in entries.items()
        )
        with self._env_path.open("a", encoding="utf-8", newline="") as handle:
            if prefix:
                handle.write(prefix)
            handle.write(text)
            handle.write(line_sep)

    # ------------------------------------------------------------------
    def _is_defined(self, name: str) -> bool:
        current = self._environ.get(name)
        if current:
            self._file_values.setdefault(name, current)
            return True
        if name in self._file_values and self._file_values[name]:
            self._environ.setdefault(name, self._file_values[name])
            return True
        return False

    # ------------------------------------------------------------------
    def _generate_value(self, name: str) -> str:
        generator = self._generators().get(name, self._fallback_generator)
        return generator()

    # ------------------------------------------------------------------
    def _generators(self) -> dict[str, Callable[[], str]]:
        return {
            "DATABASE_URL": self._default_database_url,
            "OPENAI_API_KEY": lambda: self._random_token("openai"),
            "MENACE_EMAIL": self._default_email,
            "MENACE_PASSWORD": lambda: self._random_password(24),
            "POSTGRES_USER": lambda: "menace",
            "POSTGRES_PASSWORD": lambda: self._random_password(20),
        }

    # ------------------------------------------------------------------
    def _fallback_generator(self) -> str:
        return self._random_token("menace")

    # ------------------------------------------------------------------
    def _random_token(self, prefix: str, length: int = 32) -> str:
        token = secrets.token_urlsafe(length)
        return f"{prefix}-{token}"

    # ------------------------------------------------------------------
    def _default_database_url(self) -> str:
        data_root = (self._repo_root / "sandbox_data").resolve()
        try:
            data_root.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - best effort
            data_root = self._repo_root.resolve()
        db_path = (data_root / "menace.db").resolve()
        return f"sqlite:///{db_path.as_posix()}"

    # ------------------------------------------------------------------
    def _default_email(self) -> str:
        user = getpass.getuser() or "menace"
        host = platform.node() or "sandbox"
        slug = re.sub(r"[^a-z0-9]+", ".", f"{user}.{host}".lower()).strip(".")
        slug = slug or "menace.sandbox"
        return f"{slug}@example.com"

    # ------------------------------------------------------------------
    def _random_password(self, length: int) -> str:
        alphabet = string.ascii_letters + string.digits
        specials = "!@#$%^&*-_"
        pool = alphabet + specials
        while True:
            candidate = "".join(secrets.choice(pool) for _ in range(length))
            if (
                any(c.islower() for c in candidate)
                and any(c.isupper() for c in candidate)
                and any(c.isdigit() for c in candidate)
                and any(c in specials for c in candidate)
            ):
                return candidate

    # ------------------------------------------------------------------
    def _escape(self, value: str) -> str:
        if not value:
            return ""
        if any(ch.isspace() for ch in value) or any(ch in value for ch in {'"', "'", "#"}):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value

    # ------------------------------------------------------------------
    def _redact(self, key: str) -> str:
        return f"{key}=***" if key in _SENSITIVE_KEYS else key


def ensure_bootstrap_defaults(
    required: Iterable[str],
    *,
    repo_root: Path | None = None,
    env_file: str | os.PathLike[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[set[str], Path]:
    """Populate defaults for ``required`` environment variables."""

    manager = EnvironmentDefaultsManager(
        repo_root=repo_root, env_file=env_file, environ=environ
    )
    created = manager.ensure(required)
    return created, manager.env_path


__all__ = ["EnvironmentDefaultsManager", "ensure_bootstrap_defaults"]
