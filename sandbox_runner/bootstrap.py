from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

from packaging.version import Version

from menace.auto_env_setup import ensure_env
from menace.default_config_manager import DefaultConfigManager
from sandbox_settings import SandboxSettings, load_sandbox_settings

from .cli import main as _cli_main
from .cycle import ensure_vector_service


logger = logging.getLogger(__name__)


def _ensure_sqlite_db(path: Path) -> None:
    """Ensure an SQLite database exists at ``path``."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def initialize_autonomous_sandbox(
    settings: SandboxSettings | None = None,
) -> SandboxSettings:
    """Prepare data directories, baseline metrics and optional services.

    The helper creates ``sandbox_data`` and baseline metric files when missing,
    initialises a couple of lightweight SQLite databases and verifies that the
    optional ``relevancy_radar`` and ``quick_fix_engine`` modules are available.
    A :class:`SandboxSettings` instance is returned for convenience.
    """

    settings = settings or load_sandbox_settings()

    # Populate environment defaults without prompting the user.  This creates
    # a minimal ``.env`` file when missing and exports essential configuration
    # variables to the process environment.
    try:
        DefaultConfigManager(getattr(settings, "menace_env_file", ".env")).apply_defaults()
    except Exception:  # pragma: no cover - best effort
        logger.warning("failed to ensure default configuration", exc_info=True)

    # Ensure the mandatory vector_service dependency is available before
    # proceeding with further sandbox initialisation.
    ensure_vector_service()

    data_dir = Path(settings.sandbox_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe = data_dir / ".write-test"
        probe.touch()
        probe.unlink()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("sandbox data directory %s is not writable", data_dir, exc_info=True)
        raise RuntimeError(
            f"sandbox data directory '{data_dir}' is not writable; "
            "adjust permissions or update sandbox_data_dir"
        ) from exc

    # Ensure baseline metrics file exists; fall back to minimal snapshot when
    # metrics collection fails.
    baseline_path = Path(getattr(settings, "alignment_baseline_metrics_path", ""))
    if baseline_path and not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        try:  # compute baseline if possible
            from self_improvement.metrics import _update_alignment_baseline

            _update_alignment_baseline(settings)
        except Exception:  # pragma: no cover - best effort
            logger.warning("failed to populate baseline metrics", exc_info=True)
            baseline_path.write_text("{}\n", encoding="utf-8")

    # Create expected SQLite databases
    for name in ("metrics.db", "patch_history.db", "visual_agent_queue.db"):
        _ensure_sqlite_db(data_dir / name)

    # Verify optional services are importable and meet version requirements
    from importlib import metadata

    for mod, min_version in settings.optional_service_versions.items():
        try:
            module = importlib.import_module(mod)
        except ModuleNotFoundError:
            logger.warning(
                "%s service not found; install with 'pip install %s>=%s' to enable it",
                mod,
                mod,
                min_version,
            )
            continue
        version: str | None
        try:
            version = metadata.version(mod)
        except Exception:
            version = getattr(module, "__version__", None)
        if version is None:
            logger.warning(
                "unable to determine %s version; reinstall with 'pip install --upgrade %s'",
                mod,
                mod,
            )
            continue
        if Version(version) < Version(min_version):
            logger.warning(
                "%s service version %s is too old; install %s>=%s with 'pip install --upgrade %s'",
                mod,
                version,
                mod,
                min_version,
                mod,
            )

    return settings


def _verify_required_dependencies(settings: SandboxSettings) -> dict[str, list[str]]:
    """Validate required dependencies and return missing categories.

    The function performs a pure validation pass without attempting to install
    any packages.  A dictionary mapping dependency categories to lists of missing
    packages is returned.  The categories are ``system``, ``python`` and
    ``optional``.  Callers may surface user-facing messages or decide how to
    handle the missing dependencies.
    """

    def _have_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return name in sys.modules

    def _clean_list(name: str, items: list[str]) -> list[str]:
        valid: list[str] = []
        invalid: list[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                valid.append(item.strip())
            else:
                invalid.append(str(item))
        if invalid:
            logger.warning(
                "Ignoring unrecognised %s entries: %s", name, ", ".join(invalid)
            )
        return valid

    req_tools = _clean_list("system tool", settings.required_system_tools)
    req_pkgs = _clean_list("python package", settings.required_python_packages)
    opt_pkgs = _clean_list("optional python package", settings.optional_python_packages)

    missing_sys = [t for t in req_tools if shutil.which(t) is None]
    missing_req = [p for p in req_pkgs if not _have_spec(p)]
    missing_opt = [p for p in opt_pkgs if not _have_spec(p)]

    mode = settings.menace_mode.lower()

    errors: dict[str, list[str]] = {}
    if missing_sys:
        errors["system"] = missing_sys
    if missing_req:
        errors["python"] = missing_req
    if missing_opt and mode == "production":
        errors["optional"] = missing_opt
    elif missing_opt:
        logger.warning(
            "Missing optional Python packages: %s", ", ".join(missing_opt)
        )

    return errors


def bootstrap_environment(
    settings: SandboxSettings | None = None,
    verifier: Callable[..., dict[str, list[str]]] | None = None,
    *,
    auto_install: bool = True,
) -> SandboxSettings:
    """Fully prepare the autonomous sandbox environment.

    This convenience routine ensures the environment file exists, verifies
    required dependencies, creates core SQLite databases and checks for
    optional service modules.
    """

    settings = settings or load_sandbox_settings()
    env_file = Path(settings.menace_env_file)
    created_env = not env_file.exists()
    ensure_env(str(env_file))
    # populate defaults for any missing configuration values without prompting
    DefaultConfigManager(str(env_file)).apply_defaults()
    if created_env:
        logger.info("created env file at %s", env_file)
    if verifier:
        try:
            errors = verifier(settings, auto_install)  # type: ignore[misc]
        except TypeError:
            errors = verifier(settings)  # type: ignore[misc]
    else:
        errors = _verify_required_dependencies(settings)
    if errors:
        messages: list[str] = []
        if errors.get("system"):
            messages.append(
                "Missing system packages: "
                + ", ".join(errors["system"])
                + ". Install them using your package manager."
            )
        if errors.get("python"):
            messages.append(
                "Missing Python packages: "
                + ", ".join(errors["python"])
                + ". Install them with 'pip install <package>'."
            )
        if errors.get("optional"):
            messages.append(
                "Missing optional Python packages: "
                + ", ".join(errors["optional"])
                + ". Install them with 'pip install <package>'."
            )
        raise SystemExit("\n".join(messages))
    return initialize_autonomous_sandbox(settings)


def launch_sandbox(
    settings: SandboxSettings | None = None,
    verifier: Callable[..., None] | None = None,
) -> None:
    """Run the sandbox runner using ``settings`` for configuration."""
    settings = bootstrap_environment(settings, verifier)
    # propagate core settings through environment variables
    os.environ.setdefault("SANDBOX_REPO_PATH", settings.sandbox_repo_path)
    os.environ.setdefault("SANDBOX_DATA_DIR", settings.sandbox_data_dir)
    _cli_main([])
