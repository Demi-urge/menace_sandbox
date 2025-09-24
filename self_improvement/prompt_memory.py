from __future__ import annotations

"""Lightweight prompt attempt logging for self-improvement workflows.

This module provides :func:`log_prompt_attempt` which appends metadata about
prompt executions to newline-delimited JSON files under the repository root.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterator
from uuid import uuid4

from filelock import FileLock
from dynamic_path_router import get_project_root, resolve_path

from sandbox_settings import SandboxSettings
from .prompt_strategy_manager import PromptStrategyManager

_settings = SandboxSettings()

try:
    _strategy_stats_path = Path(resolve_path("_strategy_stats.json"))
except FileNotFoundError:
    # The stats file is optional and may not exist in fresh checkouts.  Fall
    # back to the repository root so the module can lazily create it when
    # updates are recorded.
    _strategy_stats_path = Path(get_project_root()) / "_strategy_stats.json"
_strategy_lock = FileLock(str(_strategy_stats_path) + ".lock")


def load_strategy_roi_stats() -> Dict[str, Dict[str, float]]:
    """Return mapping of strategy identifiers to ROI statistics."""

    with _strategy_lock:
        try:
            data = json.loads(_strategy_stats_path.read_text(encoding="utf-8"))
            stats: Dict[str, Dict[str, float]] = {}
            for k, v in data.items():
                stats[str(k)] = {
                    "avg_roi": float(v.get("avg_roi", 0.0)),
                    "trials": int(v.get("trials", 0)),
                }
            return stats
        except Exception:
            return {}


def update_strategy_roi(strategy: str, roi_delta: float) -> None:
    """Update ROI statistics for ``strategy`` with ``roi_delta``."""

    with _strategy_lock:
        stats = load_strategy_roi_stats()
        rec = stats.setdefault(str(strategy), {"avg_roi": 0.0, "trials": 0})
        rec["trials"] += 1
        rec["avg_roi"] = (
            (rec["avg_roi"] * (rec["trials"] - 1)) + float(roi_delta)
        ) / rec["trials"]
        _strategy_stats_path.parent.mkdir(parents=True, exist_ok=True)
        _strategy_stats_path.write_text(json.dumps(stats), encoding="utf-8")


def load_prompt_penalties() -> Dict[str, int]:
    """Return mapping of prompt identifiers to downgrade counts."""

    return PromptStrategyManager().load_penalties()


def record_regression(prompt_id: str) -> int:
    """Increment downgrade count for ``prompt_id`` and persist it."""

    return PromptStrategyManager().record_penalty(prompt_id)


# Backwards compatible aliases for clarity ---------------------------------

def load_prompt_downgrades() -> Dict[str, int]:
    """Alias for :func:`load_prompt_penalties`."""

    return load_prompt_penalties()


def record_downgrade(prompt_id: str) -> int:
    """Alias for :func:`record_regression`."""

    return record_regression(prompt_id)


def reset_penalty(prompt_id: str) -> None:
    """Reset regression count for ``prompt_id`` to zero."""

    PromptStrategyManager().reset_penalty(prompt_id)


def _log_path(success: bool) -> Path:
    """Return the log file path based on *success* state."""

    filename = (
        _settings.prompt_success_log_path
        if success
        else _settings.prompt_failure_log_path
    )
    return Path(resolve_path(filename))


def log_prompt_attempt(
    prompt: Any,
    success: bool,
    exec_result: Any,
    roi_meta: Dict[str, Any] | None = None,
    prompt_id: str | None = None,
    failure_reason: str | None = None,
    sandbox_metrics: Dict[str, Any] | None = None,
    commit_hash: str | None = None,
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
        Optional string describing why the attempt failed.
    sandbox_metrics:
        Optional dictionary containing sandbox execution metrics. Selected
        metrics such as ``sandbox_score``, ``entropy`` and ``tests_passed`` are
        mirrored to top-level ``score_delta``, ``entropy_delta`` and
        ``test_status`` fields respectively in failure records. Any additional
        metrics are also copied to top-level keys for convenience.
    commit_hash:
        Optional commit hash associated with this prompt attempt.
    """

    metadata = getattr(prompt, "metadata", {}) if prompt is not None else {}
    target_module = None
    patch_id = None
    strategy = None
    if isinstance(metadata, dict):
        target_module = metadata.get("target_module") or metadata.get("module")
        patch_id = metadata.get("patch_id")
        strategy = metadata.get("strategy")
        if not prompt_id:
            prompt_id = metadata.get("prompt_id")

    if not prompt_id:
        prompt_id = str(uuid4())

    parts: list[str] = []
    for attr in ("system", "user"):
        val = getattr(prompt, attr, None)
        if val:
            parts.append(str(val))
    examples = list(getattr(prompt, "examples", []))
    parts.extend(str(e) for e in examples)
    raw_prompt = "\n".join(parts) if parts else (str(prompt) if prompt is not None else "")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "target_module": target_module,
        "patch_id": patch_id,
        "prompt_system": getattr(prompt, "system", ""),
        "prompt_user": getattr(prompt, "user", ""),
        "examples": examples,
        "metadata": metadata,
        "exec_result": exec_result,
        "prompt_id": prompt_id,
        "strategy": strategy,
        "failure_reason": failure_reason,
        "sandbox_metrics": sandbox_metrics,
        "raw_prompt": raw_prompt,
    }
    if commit_hash is not None:
        entry["commit_hash"] = commit_hash
    roi_delta: float | None = None
    if failure_reason is not None:
        # Treat the presence of a failure reason as an unsuccessful attempt even
        # if ``success`` was erroneously marked True by the caller.
        success = False

    if prompt_id and not success:
        record_regression(prompt_id)

    if roi_meta is not None:
        entry["roi_meta"] = roi_meta
        if roi_delta is None:
            roi_delta = roi_meta.get("roi_delta") if isinstance(roi_meta, dict) else None
        if roi_delta is None and isinstance(roi_meta, dict):
            roi_delta = roi_meta.get("roi")
    if not success:
        if sandbox_metrics:
            score = sandbox_metrics.get("sandbox_score")
            if score is not None:
                entry["score_delta"] = score
            entropy = sandbox_metrics.get("entropy")
            if entropy is not None:
                entry["entropy_delta"] = entropy
            tests_passed = sandbox_metrics.get("tests_passed")
            if tests_passed is None and "tests_failed" in sandbox_metrics:
                tests_passed = not bool(sandbox_metrics.get("tests_failed"))
            if tests_passed is not None:
                entry["test_status"] = bool(tests_passed)
            for k, v in sandbox_metrics.items():
                if k in {"sandbox_score", "entropy", "tests_passed", "tests_failed"}:
                    continue
                entry[k] = v

    if prompt_id and roi_delta is not None:
        try:
            update_strategy_roi(prompt_id, float(roi_delta))
        except Exception:
            pass

    path = _log_path(success)
    lock = FileLock(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def load_failures() -> Iterator[Dict[str, Any]]:
    """Stream records from the prompt failure log."""

    path = _log_path(False)
    if not path.exists():
        return iter(())

    def _iter() -> Iterator[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    return _iter()
