from __future__ import annotations

from pathlib import Path

from objective_guard import ObjectiveGuard
import startup_health_check as shc


def test_check_file_integrity_uses_objective_manifest_format(tmp_path: Path, monkeypatch) -> None:
    reward_file = tmp_path / "reward_dispatcher.py"
    reward_file.write_text("ORIGINAL\n", encoding="utf-8")

    guard = ObjectiveGuard(
        repo_root=tmp_path,
        protected_specs=["reward_dispatcher.py"],
        hash_specs=["reward_dispatcher.py"],
        manifest_path=tmp_path / "immutable_hashes_objective.json",
    )
    guard.write_manifest()

    reward_file.write_text("MODIFIED\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert shc.check_file_integrity("immutable_hashes_objective.json") == [
        "reward_dispatcher.py"
    ]


def test_startup_health_check_defaults_to_objective_manifest() -> None:
    assert shc.IMMUTABLE_HASHES_PATH == "immutable_hashes_objective.json"
