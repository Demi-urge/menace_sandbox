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
from dynamic_path_router import resolve_path
from .init import _repo_path

_settings = SandboxSettings()

_penalty_path = Path(resolve_path(_settings.prompt_penalty_path))
_penalty_lock = FileLock(str(_penalty_path) + ".lock")


def load_prompt_penalties() -> Dict[str, int]:
    """Return mapping of prompt identifiers to downgrade counts."""

    with _penalty_lock:
        try:
            data = json.loads(_penalty_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception:
            pass
        return {}


def record_regression(prompt_id: str) -> int:
    """Increment downgrade count for ``prompt_id`` and persist to disk."""

    with _penalty_lock:
        penalties = load_prompt_penalties()
        penalties[prompt_id] = penalties.get(prompt_id, 0) + 1
        _penalty_path.parent.mkdir(parents=True, exist_ok=True)
        _penalty_path.write_text(json.dumps(penalties), encoding="utf-8")
        return penalties[prompt_id]


# Backwards compatible aliases for clarity ---------------------------------

def load_prompt_downgrades() -> Dict[str, int]:
    """Alias for :func:`load_prompt_penalties`."""

    return load_prompt_penalties()


def record_downgrade(prompt_id: str) -> int:
    """Alias for :func:`record_regression`."""

    return record_regression(prompt_id)


def reset_penalty(prompt_id: str) -> None:
    """Reset regression count for ``prompt_id`` to zero."""

    with _penalty_lock:
        penalties = load_prompt_penalties()
        if prompt_id in penalties and penalties[prompt_id] != 0:
            penalties[prompt_id] = 0
            _penalty_path.parent.mkdir(parents=True, exist_ok=True)
            _penalty_path.write_text(json.dumps(penalties), encoding="utf-8")


def _log_path(success: bool) -> Path:
    """Return the log file path based on *success* state."""

    filename = (
        _settings.prompt_success_log_path
        if success
        else _settings.prompt_failure_log_path
    )
    return _repo_path() / filename


def log_prompt_attempt(
    prompt: Any,
    success: bool,
    exec_result: Any,
    roi_meta: Dict[str, Any] | None = None,
    prompt_id: str | None = None,
    failure_reason: str | None = None,
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
    failure_reason:
        Optional string describing why the attempt failed. Only stored for
        unsuccessful attempts.
    """

    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    target_module = None
    patch_id = None
    if isinstance(metadata, dict):
        target_module = metadata.get("target_module") or metadata.get("module")
        patch_id = metadata.get("patch_id")
        if prompt_id is None:
            prompt_id = metadata.get("prompt_id") or metadata.get("strategy")

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
    if prompt_id:
        entry["prompt_id"] = prompt_id
        if not success:
            record_regression(prompt_id)

    if roi_meta is not None:
        entry["roi_meta"] = roi_meta
    if failure_reason is not None:
        entry["failure_reason"] = failure_reason

    path = _log_path(success)
    lock = FileLock(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
