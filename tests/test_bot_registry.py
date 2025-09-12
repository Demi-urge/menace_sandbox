from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import menace_sandbox.patch_provenance as patch_provenance
from menace_sandbox.bot_registry import BotRegistry


class DummyService:
    """Minimal stand-in for :class:`PatchProvenanceService`."""

    def __init__(self, mapping: dict[int, str]):
        self.mapping = mapping
        self.db = self

    def get(self, patch_id: int):  # pragma: no cover - behaviour exercised via registry
        commit = self.mapping.get(patch_id)
        if commit is None:
            return None
        return SimpleNamespace(summary=json.dumps({"commit": commit}))


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    """Avoid invoking ``git`` during tests."""

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: b"")


def test_update_bot_fetches_missing_metadata_and_hot_swaps(monkeypatch, tmp_path):
    module_file = tmp_path / "mod.py"
    module_file.write_text("VALUE = 1\n")

    monkeypatch.setattr(
        patch_provenance,
        "PatchProvenanceService",
        lambda: DummyService({1: "commit1"}),
    )

    registry = BotRegistry()
    registry.update_bot("test", str(module_file), patch_id=1, commit=None)

    node = registry.graph.nodes["test"]
    assert node["module"] == str(module_file)
    assert node["commit"] == "commit1"


def test_update_bot_provenance_mismatch_recovery(monkeypatch, tmp_path):
    mod1 = tmp_path / "v1.py"
    mod2 = tmp_path / "v2.py"
    mod1.write_text("VAL = 1\n")
    mod2.write_text("VAL = 2\n")

    monkeypatch.setattr(
        patch_provenance,
        "PatchProvenanceService",
        lambda: DummyService({1: "c1", 2: "c2"}),
    )

    registry = BotRegistry()
    registry.update_bot("bot", str(mod1), patch_id=1, commit="c1")

    with pytest.raises(RuntimeError):
        registry.update_bot("bot", str(mod2), patch_id=2, commit="wrong")

    assert registry.graph.nodes["bot"]["module"] == str(mod1)

    registry.update_bot("bot", str(mod2), patch_id=2, commit="c2")
    assert registry.graph.nodes["bot"]["module"] == str(mod2)

