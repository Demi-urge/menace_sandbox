from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Callable

from menace.auto_env_setup import ensure_env
from sandbox_settings import SandboxSettings, load_sandbox_settings

from .cli import main as _cli_main


logger = logging.getLogger(__name__)


def _ensure_sqlite_db(path: Path) -> None:
    """Ensure an SQLite database exists at ``path``."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(path).close()


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

    data_dir = Path(settings.sandbox_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

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

    # Verify optional services are importable
    for mod in ("relevancy_radar", "quick_fix_engine"):
        if importlib.util.find_spec(mod) is None:
            raise RuntimeError(f"{mod} service is required but missing")

    return settings


# Dependency verification ----------------------------------------------------
REQUIRED_SYSTEM_TOOLS = ["ffmpeg", "tesseract", "qemu-system-x86_64"]
REQUIRED_PYTHON_PKGS = ["filelock", "pydantic", "dotenv"]
OPTIONAL_PYTHON_PKGS = [
    "matplotlib",
    "statsmodels",
    "uvicorn",
    "fastapi",
    "sklearn",
    "stripe",
    "httpx",
]


def _verify_required_dependencies(settings: SandboxSettings) -> None:
    """Exit if required or production optional dependencies are missing."""

    def _have_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return name in sys.modules

    missing_sys = [t for t in REQUIRED_SYSTEM_TOOLS if shutil.which(t) is None]
    missing_req = [p for p in REQUIRED_PYTHON_PKGS if not _have_spec(p)]
    missing_opt = [p for p in OPTIONAL_PYTHON_PKGS if not _have_spec(p)]

    mode = settings.menace_mode.lower()

    messages: list[str] = []
    if missing_sys:
        messages.append(
            "Missing system packages: "
            + ", ".join(missing_sys)
            + ". Install them using your package manager."
        )
    if missing_req:
        messages.append(
            "Missing Python packages: "
            + ", ".join(missing_req)
            + ". Install them with 'pip install <package>'."
        )
    if missing_opt and mode == "production":
        messages.append(
            "Missing optional Python packages: "
            + ", ".join(missing_opt)
            + ". Install them with 'pip install <package>'."
        )

    if messages:
        raise SystemExit("\n".join(messages))

    if missing_opt:
        logger.warning(
            "Missing optional Python packages: %s",
            ", ".join(missing_opt),
        )


def bootstrap_environment(
    settings: SandboxSettings | None = None,
    verifier: Callable[[SandboxSettings], None] | None = None,
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
    if created_env:
        logger.info("created env file at %s", env_file)
    (verifier or _verify_required_dependencies)(settings)
    return initialize_autonomous_sandbox(settings)


def launch_sandbox(settings: SandboxSettings | None = None) -> None:
    """Run the sandbox runner using ``settings`` for configuration."""
    if settings is None:
        settings = load_sandbox_settings()
    # propagate core settings through environment variables
    os.environ.setdefault("SANDBOX_REPO_PATH", settings.sandbox_repo_path)
    os.environ.setdefault("SANDBOX_DATA_DIR", settings.sandbox_data_dir)
    _cli_main([])
