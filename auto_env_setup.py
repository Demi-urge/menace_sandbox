from __future__ import annotations

"""Automatic environment setup utilities."""

import os
import logging
from pathlib import Path
from typing import Dict, Iterable

logger = logging.getLogger(__name__)

from .secrets_manager import SecretsManager
from .vault_secret_provider import VaultSecretProvider
from . import config_discovery as cd

# Default environment variables with fallbacks
# These are persisted to ``.env`` when missing so the application can
# start without manual configuration. Values can still be overridden via
# environment variables or command line flags.
DEFAULT_VARS: Dict[str, str] = {
    "DATABASE_URL": "sqlite:///menace.db",
    "MODELS": "demo",
    # delay between orchestration cycles; 0 runs continuously
    "SLEEP_SECONDS": "0",
    "AUTO_BOOTSTRAP": "1",
    "AUTO_UPDATE": "1",
    "UPDATE_INTERVAL": "86400",
    "OVERRIDE_UPDATE_INTERVAL": "600",
    "AUTO_BACKUP": "0",
    "MAINTENANCE_DB": "maintenance.db",
    "SANDBOX_DATA_DIR": "sandbox_data",
    "VISUAL_AGENT_TOKEN": "tombalolosvisualagent123",
    "VISUAL_AGENT_URLS": "http://127.0.0.1:8001",
}


def ensure_env(path: str = ".env") -> None:
    """Create ``path`` with required variables and sensible defaults.

    Missing values are generated or taken from :data:`DEFAULT_VARS` and
    persisted to the env file so subsequent runs operate without additional
    setup.
    """
    env_path = Path(path)
    existing: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip() and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    secrets = SecretsManager()
    vault: VaultSecretProvider | None = None
    if os.getenv("SECRET_VAULT_URL") or existing.get("SECRET_VAULT_URL"):
        try:
            vault = VaultSecretProvider()
        except Exception:
            vault = None

    # load known config vars
    for name in cd._DEFAULT_VARS:
        if vault:
            try:
                vault.export_env(name)
            except Exception as exc:  # log vault failures
                logger.error("failed exporting %s from vault: %s", name, exc)
        if os.getenv(name):
            val = os.environ[name]
        elif name in existing:
            val = existing[name]
        else:
            val = cd._generate_value(name)
        existing[name] = val
        os.environ[name] = val

    # base defaults
    for key, default in DEFAULT_VARS.items():
        existing.setdefault(key, default)
        os.environ.setdefault(key, existing[key])

    # Auto generate tokens for any *_KEY variables
    for key in [k for k in [*existing, *os.environ] if key_needs_secret(k)]:
        if key not in existing:
            existing[key] = secrets.get(key.lower())
            os.environ.setdefault(key, existing[key])

    # persist env and export
    env_path.write_text("\n".join(f"{k}={v}" for k, v in sorted(existing.items())))
    for k, v in existing.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("MENACE_ENV_FILE", str(env_path))

    # ensure required variables are present
    missing = [var for var in cd._DEFAULT_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {', '.join(missing)}"
        )


def key_needs_secret(name: str) -> bool:
    return name.upper().endswith("_KEY")




def interactive_setup(
    api_keys: Iterable[str] | None = None,
    *,
    secrets: SecretsManager | None = None,
    defaults_file: str | None = None,
) -> None:
    """Ensure required API keys are present.

    Missing keys are automatically retrieved from :class:`SecretsManager` and
    optional :class:`VaultSecretProvider` instances without prompting the user.
    """

    api_keys = list(api_keys or cd._DEFAULT_VARS)
    manager = secrets or SecretsManager()
    vault: VaultSecretProvider | None = None
    if os.getenv("SECRET_VAULT_URL"):
        try:
            vault = VaultSecretProvider(manager=manager)
        except Exception as exc:  # pragma: no cover - best effort
            vault = None
            logger.error("failed to init vault provider: %s", exc)

    defaults: Dict[str, str] = {}
    defaults_path = defaults_file or os.getenv("MENACE_DEFAULTS_FILE")
    if defaults_path:
        try:
            for line in Path(defaults_path).read_text().splitlines():
                if line.strip() and "=" in line:
                    k, v = line.split("=", 1)
                    defaults[k.strip()] = v.strip()
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed to read defaults file %s: %s", defaults_path, exc)

    for key in api_keys:
        if os.getenv(key):
            manager.set(key.lower(), os.environ[key])
            continue
        value = None
        if vault:
            try:
                value = vault.get(key.lower())
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("vault fetch failed for %s: %s", key, exc)
        if not value:
            value = manager.secrets.get(key.lower())
        if not value:
            value = defaults.get(key)
            if value:
                manager.set(key.lower(), value)
        if not value:
            value = manager.get(key.lower())
        os.environ[key] = value



__all__ = ["ensure_env", "key_needs_secret", "interactive_setup"]
