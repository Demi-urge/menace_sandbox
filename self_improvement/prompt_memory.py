from __future__ import annotations

"""Lightweight prompt attempt logging for self-improvement workflows.

This module provides :func:`log_prompt_attempt` which appends metadata about
prompt executions to newline-delimited JSON files under the repository root.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict

from filelock import FileLock

from sandbox_settings import SandboxSettings
from .init import _repo_path

_settings = SandboxSettings()


def _log_path(success: bool) -> Path:
    """Return the log file path based on *success* state."""

    filename = (
        _settings.prompt_success_log_path
        if success
        else _settings.prompt_failure_log_path
    )
    return _repo_path() / filename


def log_prompt_attempt(
    prompt: Any, success: bool, exec_result: Any, roi_meta: Dict[str, Any] | None = None
) -> None:
    """Record a prompt attempt outcome.

    Parameters
    ----------
    prompt:
        Prompt object or mapping containing ``system``, ``user``, ``examples`` and
        ``metadata`` attributes.  ``metadata`` may include ``target_module`` and
        ``patch_id``.
    success:
        ``True`` when the attempt was successful, ``False`` otherwise.  The flag
        selects between success and failure log files.
    exec_result:
        Execution result information to store alongside the prompt.  The value
        is serialised using :func:`json.dumps` with ``default=str`` so arbitrary
        objects can be stored.
    roi_meta:
        Optional ROI metrics or other contextual information.
    """

    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    target_module = None
    patch_id = None
    if isinstance(metadata, dict):
        target_module = metadata.get("target_module") or metadata.get("module")
        patch_id = metadata.get("patch_id")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "target_module": target_module,
        "patch_id": patch_id,
        "prompt_system": getattr(prompt, "system", ""),
        "prompt_user": getattr(prompt, "user", ""),
        "examples": list(getattr(prompt, "examples", [])),
        "metadata": metadata,
        "exec_result": exec_result,
    }
    if roi_meta is not None:
        entry["roi_meta"] = roi_meta

    path = _log_path(success)
    lock = FileLock(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
