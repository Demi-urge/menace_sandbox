from __future__ import annotations

"""Patch application helpers for the self-improvement engine.

The self-improvement engine ultimately delegates patch creation to the
``patch_generation`` module.  Exposing the helper via this dedicated module
keeps the public interface lightweight and focused on applying patches rather
than the underlying generation details.
"""

from pathlib import Path
import logging
import subprocess

from .patch_generation import generate_patch
from .utils import _load_callable, _call_with_retries
from ..sandbox_settings import SandboxSettings

_settings = SandboxSettings()


def apply_patch(
    patch_id: int,
    repo_path: str | Path,
    *,
    retries: int = _settings.patch_retries,
    delay: float = _settings.patch_retry_delay,
) -> None:
    """Fetch and apply patch ``patch_id`` to ``repo_path``.

    Raises ``RuntimeError`` if fetching or applying the patch fails.
    """

    logger = logging.getLogger(__name__)
    try:
        fetch = _load_callable("quick_fix_engine", "fetch_patch")
    except RuntimeError as exc:  # pragma: no cover - best effort logging
        logger.error("quick_fix_engine missing", exc_info=exc)
        raise RuntimeError(
            "quick_fix_engine is required for patch application. Install it via `pip install quick_fix_engine`."
        ) from exc
    try:
        patch_data = _call_with_retries(
            fetch, patch_id, retries=retries, delay=delay
        )
    except (RuntimeError, OSError) as exc:  # pragma: no cover - best effort logging
        logger.error("quick_fix_engine failed", exc_info=exc)
        raise RuntimeError("quick_fix_engine failed to fetch patch") from exc
    if not patch_data:
        logger.error("quick_fix_engine returned no patch data")
        raise RuntimeError("quick_fix_engine did not return patch data")
    try:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=patch_data,
            text=True,
            capture_output=True,
            cwd=str(repo_path),
            check=False,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.error("git apply invocation failed", exc_info=exc)
        raise RuntimeError("failed to apply patch") from exc
    if proc.returncode != 0:
        logger.error("git apply failed: %s", proc.stderr.strip())
        raise RuntimeError("patch application failed")


__all__ = ["generate_patch", "apply_patch"]
