import json
import types

import pytest

import menace_sandbox.patch_provenance as patch_provenance
from menace_sandbox.bot_registry import BotRegistry


class DummyBus:
    def __init__(self) -> None:
        self.events = []

    def publish(self, name: str, payload: dict) -> None:  # pragma: no cover - simple
        self.events.append((name, payload))


def _stub_service(monkeypatch):
    class DummyService:
        def __init__(self):
            self.db = self

        def get(self, pid):
            commit = {1: "a", 2: "b"}.get(pid)
            if commit is None:
                return None
            return types.SimpleNamespace(summary=json.dumps({"commit": commit}))

    monkeypatch.setattr(patch_provenance, "PatchProvenanceService", DummyService)


def test_manual_commit_requires_provenance(tmp_path, monkeypatch):
    _stub_service(monkeypatch)
    module_old = tmp_path / "dummy.py"
    module_old.write_text("VALUE = 1\n")
    module_new = tmp_path / "dummy_new.py"
    module_new.write_text("VALUE = 2\n")

    bus = DummyBus()
    registry = BotRegistry(event_bus=bus)

    class DummyManager:
        pass

    manager = DummyManager()
    manager._last_patch_id = 1
    manager._last_commit_hash = "a"

    registry.register_bot("dummy")
    registry.graph.nodes["dummy"]["selfcoding_manager"] = manager
    registry.update_bot("dummy", module_old.as_posix(), patch_id=1, commit="a")

    with pytest.raises(RuntimeError, match="update blocked"):
        registry.update_bot("dummy", module_new.as_posix(), patch_id=2, commit="b")

    event, payload = bus.events[-1]
    assert event == "bot:update_blocked"
    assert payload["reason"] == "unverified_provenance"
    node = registry.graph.nodes["dummy"]
    assert node["module"] == module_old.as_posix()
    assert node.get("update_blocked")
