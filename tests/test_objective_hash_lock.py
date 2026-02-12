from __future__ import annotations

from pathlib import Path

import pytest

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
from objective_hash_lock import verify_objective_hash_lock


def test_verify_objective_hash_lock_returns_current_hashes(tmp_path: Path) -> None:
    objective = tmp_path / "reward_dispatcher.py"
    objective.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )
    guard.write_manifest()

    report = verify_objective_hash_lock(guard=guard)

    assert report["manifest_path"] == "config/objective_hash_lock.json"
    assert report["current_hashes"]["reward_dispatcher.py"]


def test_verify_objective_hash_lock_exposes_current_hashes_on_mismatch(tmp_path: Path) -> None:
    objective = tmp_path / "reward_dispatcher.py"
    objective.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )
    guard.write_manifest()
    objective.write_text("MODIFIED\n", encoding="utf-8")

    with pytest.raises(ObjectiveGuardViolation) as exc_info:
        verify_objective_hash_lock(guard=guard)

    assert exc_info.value.reason == "manifest_hash_mismatch"
    assert "current_hashes" in exc_info.value.details
