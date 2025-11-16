from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import menace_sandbox.patch_provenance as patch_provenance
from menace_sandbox.bot_registry import BotRegistry, get_bot_workflow_tests
from menace_sandbox.override_validator import generate_signature
from menace_sandbox.sandbox_settings import BotThresholds, SandboxSettings


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

    key_path = tmp_path / "key"
    key_path.write_text("secret")
    data = {"patch_id": 1, "commit": "commit1"}
    sig = generate_signature(data, str(key_path))
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"data": data, "signature": sig}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", str(key_path))

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
    key_path = tmp_path / "key"
    key_path.write_text("secret")

    registry = BotRegistry()
    # first update succeeds
    data1 = {"patch_id": 1, "commit": "c1"}
    sig1 = generate_signature(data1, str(key_path))
    prov1 = tmp_path / "prov1.json"
    prov1.write_text(json.dumps({"data": data1, "signature": sig1}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov1))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", str(key_path))
    registry.update_bot("bot", str(mod1), patch_id=1, commit="c1")

    # mismatch: commit "wrong" signed correctly but conflicts with provenance DB
    data_wrong = {"patch_id": 2, "commit": "wrong"}
    sig_wrong = generate_signature(data_wrong, str(key_path))
    prov_wrong = tmp_path / "prov_wrong.json"
    prov_wrong.write_text(json.dumps({"data": data_wrong, "signature": sig_wrong}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov_wrong))
    with pytest.raises(RuntimeError):
        registry.update_bot("bot", str(mod2), patch_id=2, commit="wrong")

    assert registry.graph.nodes["bot"]["module"] == str(mod1)

    # correct update
    data2 = {"patch_id": 2, "commit": "c2"}
    sig2 = generate_signature(data2, str(key_path))
    prov2 = tmp_path / "prov2.json"
    prov2.write_text(json.dumps({"data": data2, "signature": sig2}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov2))
    registry.update_bot("bot", str(mod2), patch_id=2, commit="c2")
    assert registry.graph.nodes["bot"]["module"] == str(mod2)


def test_update_bot_verifies_signed_provenance(monkeypatch, tmp_path):
    module = tmp_path / "bot.py"
    module.write_text("VAL=1\n")
    registry = BotRegistry()
    calls: list[tuple[int, str]] = []

    monkeypatch.setattr(
        patch_provenance,
        "PatchProvenanceService",
        lambda: DummyService({7: "deadbeef"}),
    )

    def ok(self, pid, commit):
        calls.append((pid, commit))
        return True

    monkeypatch.setattr(BotRegistry, "_verify_signed_provenance", ok)
    registry.update_bot("bot", str(module), patch_id=7, commit="deadbeef")
    assert calls == [(7, "deadbeef")]


def test_update_bot_verification_failure(monkeypatch, tmp_path):
    class Bus:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def publish(self, name: str, payload: dict) -> None:
            self.events.append((name, payload))

    module = tmp_path / "bot.py"
    module.write_text("VAL=1\n")
    bus = Bus()
    registry = BotRegistry(event_bus=bus)

    def fail(self, *_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(BotRegistry, "_verify_signed_provenance", fail)
    with pytest.raises(RuntimeError, match="update blocked"):
        registry.update_bot("bot", str(module), patch_id=1, commit="abc")
    assert bus.events[-1][0] == "bot:update_blocked"
    assert bus.events[-1][1]["reason"] == "unverified_provenance"


def test_successful_patch_records_success_flag(monkeypatch, tmp_path):
    class Bus:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def publish(self, name: str, payload: dict) -> None:
            self.events.append((name, payload))

    module = tmp_path / "bot.py"
    module.write_text("VAL=1\n")
    bus = Bus()
    registry = BotRegistry(event_bus=bus)

    monkeypatch.setattr(
        BotRegistry,
        "_verify_signed_provenance",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(BotRegistry, "hot_swap_bot", lambda *_a, **_k: None)
    monkeypatch.setattr(BotRegistry, "health_check_bot", lambda *_a, **_k: None)

    registry.update_bot("bot", str(module), patch_id=42, commit="deadbeef")

    node = registry.graph.nodes["bot"]
    status = node[registry._PATCH_STATUS_KEY]["42"]
    assert status["patch_success"] is True

    update_event = next(
        payload for topic, payload in bus.events if topic == "bot:updated"
    )
    assert update_event["patch_success"] is True


def test_unsigned_update_warning_emitted_once(monkeypatch, tmp_path, caplog):
    import menace_sandbox.bot_registry as br

    with br._UNSIGNED_PROVENANCE_WARNING_LOCK:
        br._UNSIGNED_PROVENANCE_WARNING_CACHE.clear()
        br._UNSIGNED_PROVENANCE_WARNING_LAST_TS.clear()

    module = tmp_path / "bot.py"
    module.write_text("VAL=1\n")

    monkeypatch.setattr(br.BotRegistry, "hot_swap_bot", lambda *_a, **_k: None)
    monkeypatch.setattr(br.BotRegistry, "health_check_bot", lambda *_a, **_k: None)

    registry = br.BotRegistry()
    caplog.set_level("WARNING")

    registry.update_bot("bot", str(module), patch_id=-123, commit="unsigned:abc")
    first_warnings = [
        record
        for record in caplog.records
        if "Applying unsigned provenance update" in record.message
    ]
    assert len(first_warnings) == 1

    caplog.clear()
    node = registry.graph.nodes["bot"]
    node.pop(registry._PATCH_STATUS_KEY, None)

    registry.update_bot("bot", str(module), patch_id=-123, commit="unsigned:abc")

    assert not any(
        "Applying unsigned provenance update" in record.message for record in caplog.records
    )


def test_unsigned_update_warning_rate_limited(monkeypatch, tmp_path, caplog):
    import menace_sandbox.bot_registry as br

    with br._UNSIGNED_PROVENANCE_WARNING_LOCK:
        br._UNSIGNED_PROVENANCE_WARNING_CACHE.clear()
        br._UNSIGNED_PROVENANCE_WARNING_LAST_TS.clear()

    module = tmp_path / "bot.py"
    module.write_text("VAL=1\n")

    monkeypatch.setattr(br.BotRegistry, "hot_swap_bot", lambda *_a, **_k: None)
    monkeypatch.setattr(br.BotRegistry, "health_check_bot", lambda *_a, **_k: None)

    registry = br.BotRegistry()

    caplog.set_level("WARNING")

    monkeypatch.setattr(br, "_UNSIGNED_PROVENANCE_WARNING_INTERVAL_SECONDS", 60.0)

    current_time = [1_000.0]

    def fake_time() -> float:
        return current_time[0]

    monkeypatch.setattr(br.time, "time", fake_time)

    registry.update_bot("bot", str(module), patch_id=-1, commit="unsigned:abc")
    first_warnings = [
        record
        for record in caplog.records
        if "Applying unsigned provenance update" in record.message
    ]
    assert len(first_warnings) == 1

    caplog.clear()
    current_time[0] += 10.0  # below the interval
    registry.update_bot("bot", str(module), patch_id=-2, commit="unsigned:def")

    assert not any(
        "Applying unsigned provenance update" in record.message for record in caplog.records
    )

    caplog.clear()
    current_time[0] += 120.0  # beyond the interval
    registry.update_bot("bot", str(module), patch_id=-3, commit="unsigned:ghi")

    assert any(
        "Applying unsigned provenance update" in record.message for record in caplog.records
    )


def test_duplicate_commit_skips_unsigned_reapply(monkeypatch, tmp_path, caplog):
    import menace_sandbox.bot_registry as br

    with br._UNSIGNED_PROVENANCE_WARNING_LOCK:
        br._UNSIGNED_PROVENANCE_WARNING_CACHE.clear()
        br._UNSIGNED_PROVENANCE_WARNING_LAST_TS.clear()

    module = tmp_path / "bot.py"
    module.write_text("VAL=1\n")

    hot_swap_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def track_hot_swap(*args, **kwargs):
        hot_swap_calls.append((args, kwargs))

    monkeypatch.setattr(br.BotRegistry, "hot_swap_bot", track_hot_swap)
    monkeypatch.setattr(br.BotRegistry, "health_check_bot", lambda *_a, **_k: None)

    registry = br.BotRegistry()
    caplog.set_level("INFO")

    commit = "unsigned:abc"

    registry.update_bot("bot", str(module), patch_id=-1, commit=commit)
    assert len(hot_swap_calls) == 1

    caplog.clear()
    registry.update_bot("bot", str(module), patch_id=-2, commit=commit)

    assert len(hot_swap_calls) == 1, "duplicate commit should not trigger hot swap"
    assert any(
        "commit" in record.message and "already active" in record.message
        for record in caplog.records
    )

    node = registry.graph.nodes["bot"]
    alias_entry = registry._get_patch_status_entry(node, -2)
    assert alias_entry is not None
    assert alias_entry.get("status") == "applied"

def test_get_bot_workflow_tests_combines_sources(monkeypatch):
    registry = BotRegistry()
    registry.graph.add_node("ExampleBot")
    registry.graph.nodes["ExampleBot"]["workflow_pytest_args"] = [
        "tests/from_registry.py::test_flow"
    ]
    settings = SandboxSettings(
        bot_thresholds={
            "ExampleBot": BotThresholds(
                workflow_tests=["tests/from_settings.py::test_primary"]
            ),
            "default": BotThresholds(
                workflow_tests=["tests/from_settings_default.py::test_sanity"]
            ),
        }
    )

    import menace_sandbox.bot_registry as br

    def fake_load(bot: str | None, _settings=None):
        if bot == "ExampleBot":
            return SimpleNamespace(workflow_tests=["tests/from_thresholds.py"])
        return SimpleNamespace(workflow_tests=["tests/from_thresholds_default.py"])

    monkeypatch.setattr(br.threshold_service, "load", fake_load)

    combined = registry.get_workflow_tests("ExampleBot", settings=settings)
    assert combined == [
        "tests/from_registry.py::test_flow",
        "tests/from_settings.py::test_primary",
        "tests/from_settings_default.py::test_sanity",
        "tests/from_thresholds.py",
        "tests/from_thresholds_default.py",
    ]

    via_helper = get_bot_workflow_tests("ExampleBot", registry=registry, settings=settings)
    assert via_helper == combined

    without_registry = get_bot_workflow_tests("ExampleBot", settings=settings)
    assert without_registry == [
        "tests/from_settings.py::test_primary",
        "tests/from_settings_default.py::test_sanity",
        "tests/from_thresholds.py",
        "tests/from_thresholds_default.py",
    ]

