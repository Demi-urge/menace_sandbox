from __future__ import annotations

from pathlib import Path
import sys

import pytest

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
from objective_hash_lock import verify_objective_hash_lock
from tools import objective_guard_manifest_cli


def test_verify_objective_hash_lock_returns_current_hashes(tmp_path: Path) -> None:
    objective = tmp_path / "reward_dispatcher.py"
    objective.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )
    guard.write_manifest(operator="alice", reason="approved change", command_source="tools/objective_guard_manifest_cli.py")

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
    guard.write_manifest(operator="alice", reason="approved change", command_source="tools/objective_guard_manifest_cli.py")
    objective.write_text("MODIFIED\n", encoding="utf-8")

    with pytest.raises(ObjectiveGuardViolation) as exc_info:
        verify_objective_hash_lock(guard=guard)

    assert exc_info.value.reason == "manifest_hash_mismatch"
    assert "current_hashes" in exc_info.value.details


def test_manifest_refresh_only_via_manual_cli_command_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MENACE_SELF_CODING_OBJECTIVE_HASH_PATHS", "reward_dispatcher.py")
    monkeypatch.setenv("MENACE_SELF_CODING_PROTECTED_PATHS", "reward_dispatcher.py")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "objective_guard_manifest_cli.py",
            "--operator",
            "alice",
            "--reason",
            "approved update",
            "refresh",
        ],
    )
    rc = objective_guard_manifest_cli.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "updated manifest" in out

    guard = ObjectiveGuard(repo_root=tmp_path)
    with pytest.raises(ObjectiveGuardViolation) as exc_info:
        guard.write_manifest(
            operator="alice",
            reason="approved update",
            command_source="autonomous_runtime",
        )
    assert exc_info.value.reason == "manifest_refresh_manual_only"


def test_verify_objective_hash_lock_aligned_baseline(tmp_path: Path) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )
    guard.write_manifest(
        operator="alice",
        reason="approved bootstrap",
        command_source="tools/objective_guard_manifest_cli.py",
    )

    report = verify_objective_hash_lock(guard=guard)
    assert report["files"] == ["reward_dispatcher.py"]


def test_verify_objective_hash_lock_detects_file_set_drift(tmp_path: Path) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    kpi = tmp_path / "kpi_reward_core.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")
    kpi.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py", "kpi_reward_core.py"],
        hash_specs=["reward_dispatcher.py", "kpi_reward_core.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )
    guard.write_manifest(
        operator="alice",
        reason="approved bootstrap",
        command_source="tools/objective_guard_manifest_cli.py",
    )

    drifted_guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py", "kpi_reward_core.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )

    with pytest.raises(ObjectiveGuardViolation) as exc_info:
        verify_objective_hash_lock(guard=drifted_guard)

    assert exc_info.value.reason == "manifest_file_set_mismatch"
    assert exc_info.value.details["extra_in_manifest"] == ["kpi_reward_core.py"]


def test_verify_objective_hash_lock_allows_legitimate_rotation(tmp_path: Path) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config" / "objective_hash_lock.json",
    )
    guard.write_manifest(
        operator="alice",
        reason="approved bootstrap",
        command_source="tools/objective_guard_manifest_cli.py",
    )

    reward.write_text("ROTATED\n", encoding="utf-8")
    with pytest.raises(ObjectiveGuardViolation):
        verify_objective_hash_lock(guard=guard)

    guard.write_manifest(
        operator="bob",
        reason="approved objective update",
        rotation=True,
        command_source="tools/objective_guard_manifest_cli.py",
    )

    report = verify_objective_hash_lock(guard=guard)
    assert report["files"] == ["reward_dispatcher.py"]
