from __future__ import annotations

from pathlib import Path

import json

from objective_guard import ObjectiveGuard
import startup_health_check as shc


def test_check_file_integrity_uses_objective_manifest_format(tmp_path: Path, monkeypatch) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "config/objective_hash_lock.json",
    )
    guard.write_manifest(operator="alice", reason="approved change", command_source="tools/objective_guard_manifest_cli.py")
    manifest_path = tmp_path / "config/objective_hash_lock.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"]["config/objective_hash_lock.json"] = payload["files"].pop("reward_dispatcher.py")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    reward_file.write_text("MODIFIED\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    mismatches = shc.check_file_integrity("config/objective_hash_lock.json")
    assert "reward_dispatcher.py" in mismatches


def test_startup_health_check_defaults_to_objective_manifest() -> None:
    assert shc.IMMUTABLE_HASHES_PATH == "config/objective_hash_lock.json"
