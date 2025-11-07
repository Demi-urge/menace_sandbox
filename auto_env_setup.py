from __future__ import annotations

"""Automatic environment setup utilities.

Stripe API keys are intentionally excluded from generated configuration;
``stripe_billing_router`` reads them directly from the environment to avoid
writing sensitive values to disk."""

import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

if __package__ in {None, ""}:
    # Support running as a standalone script by ensuring the package root is importable.
    package_root = Path(__file__).resolve().parent
    package_parent = str(package_root.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import menace_sandbox.config_discovery as cd
    from menace_sandbox.secrets_manager import SecretsManager
    from menace_sandbox.vault_secret_provider import VaultSecretProvider
    from menace_sandbox.dynamic_path_router import resolve_path
else:
    from . import config_discovery as cd
    from .secrets_manager import SecretsManager
    from .vault_secret_provider import VaultSecretProvider
    from .dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)

try:
    _DEFAULT_SANDBOX_DATA_DIR = str(resolve_path("sandbox_data"))
except FileNotFoundError as exc:
    fallback_dir = Path(__file__).resolve().parent / "sandbox_data"
    try:
        fallback_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Directory creation is best-effort; use parent as final fallback
        fallback_dir = fallback_dir.parent
    logger.warning(
        "sandbox_data directory not found; using fallback path %s", fallback_dir, exc_info=exc
    )
    _DEFAULT_SANDBOX_DATA_DIR = str(fallback_dir)

# Shared defaults
RECURSIVE_ISOLATED_DEFAULT = "1"
RECURSIVE_ISOLATED_VARS = (
    "SANDBOX_RECURSIVE_ISOLATED",
    "SELF_TEST_RECURSIVE_ISOLATED",
)

SENSITIVE_VARS = ()

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
    "SANDBOX_DATA_DIR": _DEFAULT_SANDBOX_DATA_DIR,
    "SELF_TEST_DISABLE_ORPHANS": "0",
    "SELF_TEST_DISCOVER_ORPHANS": "1",
    "SELF_TEST_RECURSIVE_ORPHANS": "1",
    "SANDBOX_RECURSIVE_ORPHANS": "1",
    **{var: RECURSIVE_ISOLATED_DEFAULT for var in RECURSIVE_ISOLATED_VARS},
    "SELF_TEST_AUTO_INCLUDE_ISOLATED": "1",
    "SANDBOX_AUTO_INCLUDE_ISOLATED": "1",
    "SANDBOX_TEST_REDUNDANT": "1",
}


def get_recursive_isolated() -> bool:
    """Return whether isolated module dependencies should be traversed."""
    val = (
        os.getenv("SANDBOX_RECURSIVE_ISOLATED")
        or os.getenv("SELF_TEST_RECURSIVE_ISOLATED")
        or RECURSIVE_ISOLATED_DEFAULT
    )
    return val.lower() not in {"0", "false", "no"}


def set_recursive_isolated(enabled: bool) -> None:
    """Synchronise recursion flags for isolated modules."""
    val = "1" if enabled else "0"
    os.environ["SANDBOX_RECURSIVE_ISOLATED"] = val
    os.environ["SELF_TEST_RECURSIVE_ISOLATED"] = val


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

    # merge defaults file when provided
    defaults_path = os.getenv("MENACE_DEFAULTS_FILE") or existing.get("MENACE_DEFAULTS_FILE")
    if defaults_path:
        try:
            for line in Path(defaults_path).read_text().splitlines():
                if line.strip() and "=" in line:
                    k, v = line.split("=", 1)
                    existing.setdefault(k.strip(), v.strip())
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed to read defaults file %s: %s", defaults_path, exc)

    # pull values from previous sandbox runs
    data_dir = Path(existing.get("SANDBOX_DATA_DIR", DEFAULT_VARS["SANDBOX_DATA_DIR"]))
    if data_dir.exists():
        hist_file = data_dir / "roi_history.json"
        try:
            if hist_file.exists():
                hist = json.loads(hist_file.read_text())
                if isinstance(hist, list) and hist:
                    existing.setdefault("ROI_THRESHOLD", str(hist[-1]))
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed to load ROI history %s: %s", hist_file, exc)
        try:
            preset_files = sorted(data_dir.glob("*preset*.json"))
            if preset_files:
                data = json.loads(preset_files[-1].read_text())
                existing.setdefault("SANDBOX_ENV_PRESETS", json.dumps(data))
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("failed to load preset file from %s: %s", data_dir, exc)

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

    # fetch sensitive vars from secrets manager or environment
    for key in SENSITIVE_VARS:
        if key in os.environ:
            existing.setdefault(key, os.environ[key])
        elif key not in existing:
            existing[key] = secrets.get(key.lower()) or ""
        os.environ.setdefault(key, existing.get(key, ""))

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

    # answers pre-filled via environment variables
    env_answers = {
        key: os.getenv(f"MENACE_SETUP_{key}") for key in api_keys
    }

    missing: list[str] = []
    for key in api_keys:
        if os.getenv(key):
            manager.set(key.lower(), os.environ[key])
            continue
        value = env_answers.get(key)
        if not value and vault:
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
        if value:
            os.environ[key] = value
            continue
        missing.append(key)

    interactive = os.getenv("MENACE_NON_INTERACTIVE") != "1" and sys.stdin.isatty()
    for key in missing:
        value = env_answers.get(key)
        if value is None and interactive:
            try:
                value = input(f"{key}: ").strip()
            except EOFError:
                value = ""
        if not value:
            value = manager.get(key.lower())
        else:
            manager.set(key.lower(), value)
        os.environ[key] = value


__all__ = ["ensure_env", "key_needs_secret", "interactive_setup"]
