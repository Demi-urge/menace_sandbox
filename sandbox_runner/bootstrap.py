from __future__ import annotations
from __future__ import annotations

import importlib.util

import os
import sqlite3
from pathlib import Path
import logging

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


def launch_sandbox(settings: SandboxSettings | None = None) -> None:
    """Run the sandbox runner using ``settings`` for configuration."""
    if settings is None:
        settings = load_sandbox_settings()
    # propagate core settings through environment variables
    os.environ.setdefault("SANDBOX_REPO_PATH", settings.sandbox_repo_path)
    os.environ.setdefault("SANDBOX_DATA_DIR", settings.sandbox_data_dir)
    _cli_main([])
