from __future__ import annotations

import json
from pathlib import Path

import pytest

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
from objective_surface_policy import (
    OBJECTIVE_ADJACENT_HASH_PATHS,
    OBJECTIVE_ADJACENT_UNSAFE_PATHS,
)


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
    expected = {path.rstrip("/") for path in OBJECTIVE_ADJACENT_UNSAFE_PATHS}
    assert expected.issubset(protected)



def test_objective_guard_blocks_directory_protected_target(tmp_path: Path) -> None:
    payout_file = tmp_path / "finance_logs" / "payout_log.json"
    payout_file.parent.mkdir(parents=True, exist_ok=True)
    payout_file.write_text("[]\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["finance_logs/"],
        hash_specs=[],
    )

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.assert_patch_target_safe(payout_file)

    assert exc.value.reason == "protected_target"


def test_objective_guard_defaults_apply_without_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MENACE_SELF_CODING_PROTECTED_PATHS", raising=False)
    monkeypatch.delenv("MENACE_SELF_CODING_OBJECTIVE_HASH_PATHS", raising=False)

    payout_file = tmp_path / "finance_logs" / "payout_log.json"
    payout_file.parent.mkdir(parents=True, exist_ok=True)
    payout_file.write_text("[]\n", encoding="utf-8")

    guard = ObjectiveGuard(repo_root=tmp_path)

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.assert_patch_target_safe(payout_file)

    assert exc.value.reason == "protected_target"


def test_objective_guard_manifest_hash_mismatch_exposes_changed_files(tmp_path: Path) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    ledger = tmp_path / "reward_ledger.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")
    ledger.write_text("ORIGINAL\n", encoding="utf-8")
    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py", "reward_ledger.py"],
        hash_specs=["reward_dispatcher.py", "reward_ledger.py"],
        manifest_path=tmp_path / ".security" / "state" / "objective_guard_manifest.json",
    )
    guard.write_manifest()
    reward.write_text("MODIFIED\n", encoding="utf-8")

    with pytest.raises(ObjectiveGuardViolation) as exc:
        guard.assert_integrity()

    assert exc.value.reason == "manifest_hash_mismatch"
    assert "reward_dispatcher.py" in (exc.value.details.get("changed_files") or [])


def test_objective_guard_manifest_bootstrap_records_audit_metadata(tmp_path: Path) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")

    manifest_path = tmp_path / "config" / "objective_hash_lock.json"
    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=manifest_path,
    )

    guard.write_manifest(operator="alice", reason="initial trust bootstrap")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    baseline = payload.get("trusted_baseline", {})
    assert baseline.get("mode") == "bootstrap"
    assert baseline.get("who") == "alice"
    assert baseline.get("why") == "initial trust bootstrap"
    assert isinstance(payload.get("manifest_sha256"), str)
    assert payload.get("audit", [{}])[-1].get("action") == "bootstrap"


def test_objective_guard_manifest_rotate_preserves_audit_history(tmp_path: Path) -> None:
    reward = tmp_path / "reward_dispatcher.py"
    reward.write_text("ORIGINAL\n", encoding="utf-8")

    manifest_path = tmp_path / "config" / "objective_hash_lock.json"
    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=manifest_path,
    )

    guard.write_manifest(operator="alice", reason="bootstrap baseline")
    first_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    reward.write_text("ROTATED\n", encoding="utf-8")
    guard.write_manifest(operator="bob", reason="approved objective update", rotation=True)
    second_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert second_payload["trusted_baseline"]["mode"] == "rotate"
    assert second_payload["trusted_baseline"]["who"] == "bob"
    assert second_payload["trusted_baseline"]["why"] == "approved objective update"
    assert second_payload["trusted_baseline"]["previous_manifest_sha256"] == first_payload.get("manifest_sha256")
    actions = [entry.get("action") for entry in second_payload.get("audit", [])]
    assert actions == ["bootstrap", "rotate"]


def test_objective_guard_default_hash_specs_match_canonical_hash_inventory(tmp_path: Path) -> None:
    for rel in OBJECTIVE_ADJACENT_HASH_PATHS:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")

    guard = ObjectiveGuard(repo_root=tmp_path)

    default_hashes = {spec.normalized for spec in guard.hash_specs if not spec.prefix}
    assert default_hashes == set(OBJECTIVE_ADJACENT_HASH_PATHS)


def test_objective_guard_defaults_hash_every_non_directory_protected_path(tmp_path: Path) -> None:
    for rel in OBJECTIVE_ADJACENT_UNSAFE_PATHS:
        target = tmp_path / rel.rstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.write_text("x\n", encoding="utf-8")

    guard = ObjectiveGuard(repo_root=tmp_path)

    protected_files = {spec.normalized for spec in guard.protected_specs if not spec.prefix}
    hashed_files = {spec.normalized for spec in guard.hash_specs if not spec.prefix}
    assert protected_files == hashed_files
