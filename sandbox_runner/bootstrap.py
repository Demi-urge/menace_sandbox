from __future__ import annotations

import os

from sandbox_settings import SandboxSettings, load_sandbox_settings

from .cli import main as _cli_main


def launch_sandbox(settings: SandboxSettings | None = None) -> None:
    """Run the sandbox runner using ``settings`` for configuration."""
    if settings is None:
        settings = load_sandbox_settings()
    # propagate core settings through environment variables
    os.environ.setdefault("SANDBOX_REPO_PATH", settings.sandbox_repo_path)
    os.environ.setdefault("SANDBOX_DATA_DIR", settings.sandbox_data_dir)
    _cli_main([])
