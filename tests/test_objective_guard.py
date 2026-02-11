from __future__ import annotations

from pathlib import Path

import pytest

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation


def test_objective_guard_blocks_protected_target(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("print('reward')\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
    )

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.assert_patch_target_safe(reward_file)

    assert exc.value.reason == "protected_target"


def test_objective_guard_detects_hash_drift(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
    )

    reward_file.write_text("MODIFIED\n", encoding="utf-8")

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.assert_integrity()

    assert exc.value.reason == "objective_integrity_breach"
    changed = exc.value.details.get("changed_files", [])
    assert "reward_dispatcher.py" in changed


def test_objective_guard_allows_non_protected_target(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("reward\n", encoding="utf-8")
    safe_file = tmp_path / "safe_module.py"
    safe_file.write_text("ok\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
    )

    guard.assert_integrity()
    guard.assert_patch_target_safe(safe_file)
