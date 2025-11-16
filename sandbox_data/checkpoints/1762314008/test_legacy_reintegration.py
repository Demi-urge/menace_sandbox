import types
from pathlib import Path
import logging

import pytest

import sandbox_runner.cycle as cycle
from sandbox_settings import SandboxSettings


def _dummy_metrics(monkeypatch, initial_legacy=0):
    class DummyMetric:
        def __init__(self, val=0):
            self.value = val

        def inc(self, v=1):
            self.value += v

        def dec(self, v=1):
            self.value -= v

    for name in [
        "orphan_modules_tested_total",
        "orphan_modules_reintroduced_total",
        "orphan_modules_failed_total",
        "orphan_modules_redundant_total",
        "orphan_modules_legacy_total",
        "orphan_modules_reclassified_total",
    ]:
        val = initial_legacy if name == "orphan_modules_legacy_total" else 0
        monkeypatch.setattr(cycle, name, DummyMetric(val), raising=False)


def _common_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_TEST_REDUNDANT", "1")
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "append_orphan_classifications", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "load_orphan_cache", lambda *_a, **_k: {})
    (tmp_path / "foo.py").write_text("pass\n")  # path-ignore
    def fake_discover(repo_path):
        assert Path(repo_path) == tmp_path
        return {"foo": {"parents": [], "classification": "legacy", "redundant": True}}
    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover)


def test_legacy_module_logged_and_counted(monkeypatch, tmp_path, caplog):
    calls: list[list[str]] = []
    def fake_auto_include(mods, recursive=False, validate=False):
        calls.append(list(mods))
        return object(), {"added": [], "failed": [], "redundant": list(mods)}
    _dummy_metrics(monkeypatch)
    _common_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    ctx = types.SimpleNamespace(
        repo=tmp_path,
        module_map=set(),
        orphan_traces={},
        tracker=types.SimpleNamespace(merge_history=lambda *a, **k: None),
        settings=SandboxSettings(),
    )
    caplog.set_level(logging.INFO, logger=cycle.logger.name)
    cycle.include_orphan_modules(ctx)
    assert calls == [["foo.py"]]  # path-ignore
    records = [r for r in caplog.records if r.message == "isolated module tests"]
    assert records and records[-1].legacy == ["foo.py"]  # path-ignore
    assert cycle.orphan_modules_legacy_total.value == 1


def test_legacy_module_reintroduced_decrements(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    def fake_auto_include(mods, recursive=False, validate=False):
        calls.append(list(mods))
        return object(), {"added": list(mods), "failed": [], "redundant": []}
    _dummy_metrics(monkeypatch, initial_legacy=1)
    _common_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    ctx = types.SimpleNamespace(
        repo=tmp_path,
        module_map=set(),
        orphan_traces={},
        tracker=types.SimpleNamespace(merge_history=lambda *a, **k: None),
        settings=SandboxSettings(),
    )
    cycle.include_orphan_modules(ctx)
    assert calls == [["foo.py"]]  # path-ignore
    assert cycle.orphan_modules_legacy_total.value == 0
    assert "foo.py" not in ctx.orphan_traces  # path-ignore
