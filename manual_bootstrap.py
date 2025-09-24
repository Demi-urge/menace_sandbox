"""Manual bootstrap helper for preparing the sandbox environment.

This script consolidates the two bootstrap phases that are normally handled
by the automation tooling:

* :func:`sandbox_runner.bootstrap.bootstrap_environment` ensures the sandbox
  specific configuration files, SQLite databases and optional services are in
  place.
* :class:`environment_bootstrap.EnvironmentBootstrapper` verifies external
  dependencies such as system packages, secrets and optional background
  services.

The helper keeps both steps accessible when automation is unavailable.  It can
be executed directly (``python manual_bootstrap.py``) or imported and invoked
via :func:`main`.  All heavy lifting remains in the existing bootstrap modules;
this wrapper simply wires them together with a small CLI and friendlier logging.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import MutableMapping, Sequence


def _ensure_package_on_path() -> Path:
    """Return the repository root and ensure the package is importable."""

    script_path = Path(__file__).resolve()
    package_root = script_path.parent
    marker = package_root / "__init__.py"
    if marker.exists():
        parent = package_root.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
    else:  # pragma: no cover - fallback for unusual layouts
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    return package_root


_REPO_ROOT = _ensure_package_on_path()
_DEFAULT_DATA_DIR = _REPO_ROOT / "sandbox_data"

_DYNAMIC_PATH_ROUTER = importlib.import_module("menace_sandbox.dynamic_path_router")
sys.modules["dynamic_path_router"] = _DYNAMIC_PATH_ROUTER

for _alias in ("unified_event_bus", "resilience"):
    try:
        module = importlib.import_module(f"menace_sandbox.{_alias}")
    except ModuleNotFoundError:  # pragma: no cover - optional alias
        continue
    sys.modules[_alias] = module

_menace_pkg = importlib.import_module("menace")
if not getattr(_menace_pkg, "RAISE_ERRORS", None):  # pragma: no cover - defensive reload
    _menace_pkg = importlib.reload(_menace_pkg)

from menace_sandbox.environment_bootstrap import EnvironmentBootstrapper
from menace_sandbox.sandbox_runner.bootstrap import bootstrap_environment
from menace_sandbox.sandbox_settings import SandboxSettings


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the manual bootstrap helper."""

    parser = argparse.ArgumentParser(description="Run Menace sandbox bootstrap tasks manually")
    parser.add_argument(
        "--skip-sandbox",
        action="store_true",
        help="Skip sandbox-specific checks (bootstrap_environment)",
    )
    parser.add_argument(
        "--skip-environment",
        action="store_true",
        help="Skip system/environment bootstrap (EnvironmentBootstrapper)",
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Allow sandbox bootstrap to install missing Python packages",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> logging.Logger:
    """Configure logging and return a module logger."""

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=numeric_level)
    else:
        root.setLevel(numeric_level)
    return logging.getLogger("manual_bootstrap")


def _configure_environment(
    repo_root: Path,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Ensure key environment variables point at *repo_root*."""

    target = env if env is not None else os.environ
    target.setdefault("SANDBOX_REPO_PATH", str(repo_root))
    target.setdefault("SANDBOX_DATA_DIR", str(_DEFAULT_DATA_DIR))


def _describe_settings(logger: logging.Logger, settings: SandboxSettings) -> None:
    """Emit a short summary for the provided sandbox *settings*."""

    try:
        env_file = Path(settings.menace_env_file)
    except Exception:  # pragma: no cover - defensive guard
        env_file = Path(".env")
    logger.info(
        "Sandbox configuration ready (repo=%s data_dir=%s env_file=%s)",
        getattr(settings, "sandbox_repo_path", "<unknown>"),
        getattr(settings, "sandbox_data_dir", "<unknown>"),
        env_file.resolve(),
    )


def main(
    argv: Sequence[str] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> int:
    """Execute the manual bootstrap routine.

    Parameters
    ----------
    argv:
        Optional command line arguments.  When ``None`` (the default) they are
        pulled from :data:`sys.argv`.
    env:
        Optional environment mapping.  When ``None`` the real process
        environment is used.

    Returns
    -------
    int
        ``0`` on success, otherwise a non-zero exit status describing the error.
    """

    args = _parse_args(argv)
    logger = _setup_logging(args.log_level)
    _configure_environment(_REPO_ROOT, env)

    exit_code = 0
    settings: SandboxSettings | None = None

    if not args.skip_sandbox:
        try:
            settings = bootstrap_environment(auto_install=args.auto_install)
        except SystemExit as exc:
            message = str(exc) or "sandbox dependency verification failed"
            logger.error(message)
            code = exc.code if isinstance(exc.code, int) else 1
            return code or 1
        except KeyboardInterrupt:
            logger.info("sandbox bootstrap interrupted")
            return 130
        except Exception:
            logger.exception("sandbox bootstrap failed")
            return 1
        else:
            if settings is not None:
                _describe_settings(logger, settings)
            logger.info("sandbox bootstrap completed successfully")
    else:
        logger.info("sandbox bootstrap completed successfully (skipped)")

    if args.skip_environment:
        return exit_code

    try:
        EnvironmentBootstrapper().bootstrap()
    except KeyboardInterrupt:
        logger.info("environment bootstrap interrupted")
        exit_code = 130
    except Exception:
        logger.exception("environment bootstrap failed")
        return 1
    else:
        logger.info("environment bootstrap completed successfully")

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
