from __future__ import annotations

from pathlib import Path

import pytest

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
from self_coding_objective_paths import OBJECTIVE_ADJACENT_UNSAFE_PATHS


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


def test_objective_guard_manifest_matches_hashes(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / ".security" / "state" / "objective_guard_manifest.json",
    )
    guard.write_manifest()

    report = guard.verify_manifest()

    assert report["files"] == ["reward_dispatcher.py"]


def test_objective_guard_manifest_detects_hash_mismatch(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / ".security" / "state" / "objective_guard_manifest.json",
    )
    guard.write_manifest()

    reward_file.write_text("MODIFIED\n", encoding="utf-8")

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.verify_manifest()

    assert exc.value.reason == "manifest_hash_mismatch"
    deltas = exc.value.details.get("deltas", [])
    assert deltas and deltas[0]["path"] == "reward_dispatcher.py"


def test_objective_guard_missing_manifest_fails(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("reward\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / ".security" / "state" / "objective_guard_manifest.json",
    )

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.verify_manifest()

    assert exc.value.reason == "manifest_missing"


def test_objective_guard_detects_hash_drift(tmp_path: Path) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / ".security" / "state" / "objective_guard_manifest.json",
    )
    guard.write_manifest()

    reward_file.write_text("MODIFIED\n", encoding="utf-8")

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.assert_integrity()

    assert exc.value.reason == "manifest_hash_mismatch"
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
        manifest_path=tmp_path / ".security" / "state" / "objective_guard_manifest.json",
    )
    guard.write_manifest()

    guard.assert_integrity()
    guard.assert_patch_target_safe(safe_file)


def test_objective_guard_defaults_include_shared_objective_paths(tmp_path: Path) -> None:
    for rel in OBJECTIVE_ADJACENT_UNSAFE_PATHS:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")

    guard = ObjectiveGuard(repo_root=tmp_path)

    protected = {spec.normalized for spec in guard.protected_specs}
    assert set(OBJECTIVE_ADJACENT_UNSAFE_PATHS).issubset(protected)
