from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

import repair_loop


def test_discover_repo_root_prefers_git_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    nested = repo / "pkg" / "module.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("print('hi')")

    assert repair_loop._discover_repo_root(nested, None) == repo.resolve()


def test_run_repair_loop_validates_and_applies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    target = repo / "module.py"
    target.write_text("print('hi')\n")

    calls: dict[str, dict[str, object]] = {}

    def fake_validate(**kwargs: object) -> tuple[bool, list[str]]:
        calls["validate"] = kwargs
        return True, []

    def fake_apply(**kwargs: object) -> tuple[bool, int | None, list[str]]:
        calls["apply"] = kwargs
        return True, 1, []

    builder = types.SimpleNamespace(provenance_token="token")
    monkeypatch.setattr(repair_loop, "create_context_builder", lambda repo_root: builder)
    monkeypatch.setattr(repair_loop.quick_fix, "validate_patch", fake_validate)
    monkeypatch.setattr(repair_loop.quick_fix, "apply_validated_patch", fake_apply)

    class Manager:
        def __init__(self) -> None:
            self.refreshed = False

        def refresh_quick_fix_context(self) -> None:
            self.refreshed = True

    manager = Manager()

    class Service:
        def __init__(self) -> None:
            self.manager = manager

        def run_once(self, pytest_args: list[str]):
            return types.SimpleNamespace(fail_count=0), None

    results = types.SimpleNamespace(
        diagnostics=[
            {
                "file": str(target),
                "test_name": "test_example",
                "error_summary": "boom",
            }
        ]
    )

    outcome = repair_loop.run_repair_loop(results, Service(), repo_root=repo)

    assert outcome.fail_count == 0
    assert manager.refreshed is True
    assert Path(calls["validate"]["module_path"]).resolve() == target.resolve()
    assert calls["apply"]["flags"] == []
    assert calls["apply"]["context_meta"]["repair_attempt"] == 1


def test_run_repair_loop_merges_validation_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    target = repo / "module.py"
    target.write_text("print('hi')\n")

    calls: dict[str, dict[str, object]] = {}

    def fake_validate(**kwargs: object) -> tuple[bool, list[str], dict[str, str]]:
        calls["validate"] = kwargs
        return True, [], {"from_validation": "meta"}

    def fake_apply(**kwargs: object) -> tuple[bool, int | None, list[str]]:
        calls["apply"] = kwargs
        return True, 1, []

    builder = types.SimpleNamespace(provenance_token="token")
    monkeypatch.setattr(repair_loop, "create_context_builder", lambda repo_root: builder)
    monkeypatch.setattr(repair_loop.quick_fix, "validate_patch", fake_validate)
    monkeypatch.setattr(repair_loop.quick_fix, "apply_validated_patch", fake_apply)

    manager = types.SimpleNamespace(refresh_quick_fix_context=lambda: None)

    class Service:
        def __init__(self) -> None:
            self.manager = manager

        def run_once(self, pytest_args: list[str]):
            return types.SimpleNamespace(fail_count=0), None

    results = types.SimpleNamespace(
        diagnostics=[
            {
                "file": str(target),
                "test_name": "test_example",
                "error_summary": "boom",
            }
        ]
    )

    outcome = repair_loop.run_repair_loop(results, Service(), repo_root=repo)

    assert outcome.fail_count == 0
    assert Path(calls["validate"]["module_path"]).resolve() == target.resolve()
    assert calls["apply"]["context_meta"]["from_validation"] == "meta"
    assert calls["apply"]["context_meta"]["repair_attempt"] == 1


def test_run_repair_loop_surfaces_validation_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    target = repo / "module.py"
    target.write_text("print('hi')\n")

    calls: dict[str, dict[str, object]] = {}

    def fake_validate(**kwargs: object) -> tuple[bool, list[str], dict[str, str]]:
        calls["validate"] = kwargs
        return True, ["flagged"], {"from_validation": "meta"}

    def fake_apply(**kwargs: object) -> tuple[bool, int | None, list[str]]:
        calls["apply"] = kwargs
        return False, None, list(kwargs.get("flags", []))

    builder = types.SimpleNamespace(provenance_token="token")
    monkeypatch.setattr(repair_loop, "create_context_builder", lambda repo_root: builder)
    monkeypatch.setattr(repair_loop.quick_fix, "validate_patch", fake_validate)
    monkeypatch.setattr(repair_loop.quick_fix, "apply_validated_patch", fake_apply)

    manager = types.SimpleNamespace(refresh_quick_fix_context=lambda: None)

    class Service:
        def __init__(self) -> None:
            self.manager = manager

        def run_once(self, pytest_args: list[str]):  # pragma: no cover - should not be called
            raise AssertionError("run_once should not be invoked when validation fails")

    results = types.SimpleNamespace(
        diagnostics=[
            {
                "file": str(target),
                "test_name": "test_example",
                "error_summary": "boom",
            }
        ]
    )

    with pytest.raises(repair_loop.RepairLoopError) as excinfo:
        repair_loop.run_repair_loop(results, Service(), repo_root=repo)

    assert "flagged" in str(excinfo.value)
    assert Path(calls["validate"]["module_path"]).resolve() == target.resolve()
    assert calls["apply"]["flags"] == ["flagged"]
