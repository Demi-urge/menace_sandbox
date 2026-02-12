from __future__ import annotations

"""Shared objective hash-lock verification helpers.

This module centralises objective hash-lock verification so startup checks and
per-cycle self-coding checks use the same manifest source and comparison logic.
"""

from pathlib import Path
from typing import Any

from objective_guard import (
    DEFAULT_OBJECTIVE_HASH_MANIFEST,
    ObjectiveGuard,
    ObjectiveGuardViolation,
)


def verify_objective_hash_lock(
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    guard: ObjectiveGuard | None = None,
) -> dict[str, Any]:
    """Verify objective files against the persisted SHA-256 manifest.

    Returns a report containing the current hashes and manifest metadata.
    Raises :class:`ObjectiveGuardViolation` when the manifest is missing,
    malformed, or hash deltas are detected.
    """

    root = (repo_root or Path.cwd()).resolve()
    resolved_guard = guard or ObjectiveGuard(
        repo_root=root,
        manifest_path=(manifest_path or (root / DEFAULT_OBJECTIVE_HASH_MANIFEST)).resolve(),
    )
    current_hashes = resolved_guard.snapshot_hashes()
    try:
        report = resolved_guard.verify_manifest()
    except ObjectiveGuardViolation as exc:
        details = dict(getattr(exc, "details", {}) or {})
        details.setdefault("current_hashes", current_hashes)
        raise ObjectiveGuardViolation(exc.reason, details=details) from exc

    payload: dict[str, Any] = dict(report)
    payload["current_hashes"] = current_hashes
    return payload


__all__ = ["verify_objective_hash_lock"]
