from __future__ import annotations

from pathlib import Path

from objective_guard import ObjectiveGuard
from objective_inventory import (
    CANONICAL_OBJECTIVE_INVENTORY,
    OBJECTIVE_ADJACENT_HASH_PATHS,
    OBJECTIVE_ADJACENT_UNSAFE_PATHS,
)
from self_coding_policy import get_self_coding_unsafe_path_rules


def test_canonical_objective_inventory_is_merged_into_unsafe_paths_env(monkeypatch) -> None:
    monkeypatch.setenv("MENACE_SELF_CODING_UNSAFE_PATHS", "custom_sensitive")

    merged = set(get_self_coding_unsafe_path_rules())

    assert set(OBJECTIVE_ADJACENT_UNSAFE_PATHS).issubset(merged)
    assert "custom_sensitive" in merged


def test_every_hash_lock_eligible_canonical_objective_file_is_in_hash_targets() -> None:
    expected_hash_targets = {
        item.path
        for item in CANONICAL_OBJECTIVE_INVENTORY
        if item.include_in_hash_lock
    }

    assert expected_hash_targets == set(OBJECTIVE_ADJACENT_HASH_PATHS)


def test_manifest_verification_succeeds_with_canonical_inventory_targets(tmp_path: Path) -> None:
    for rel in OBJECTIVE_ADJACENT_HASH_PATHS:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("baseline\n", encoding="utf-8")

    (tmp_path / "finance_logs").mkdir(parents=True, exist_ok=True)

    guard = ObjectiveGuard(repo_root=tmp_path)
    guard.write_manifest(
        operator="alice",
        reason="approved baseline bootstrap",
        command_source="tools/objective_guard_manifest_cli.py",
    )

    report = guard.verify_manifest()

    assert set(report["files"]) == set(OBJECTIVE_ADJACENT_HASH_PATHS)
