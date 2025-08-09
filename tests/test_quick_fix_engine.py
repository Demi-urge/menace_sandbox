import os
import sys
import types
import importlib.util
import logging
from pathlib import Path
import pytest

# Avoid heavy imports from the real package
package = types.ModuleType("menace")
sys.modules["menace"] = package

# Stub required submodules
error_bot = types.ModuleType("menace.error_bot")
error_bot.ErrorDB = object
sys.modules["menace.error_bot"] = error_bot

scm = types.ModuleType("menace.self_coding_manager")
scm.SelfCodingManager = object
sys.modules["menace.self_coding_manager"] = scm

kg = types.ModuleType("menace.knowledge_graph")
kg.KnowledgeGraph = object
sys.modules["menace.knowledge_graph"] = kg

# Load QuickFixEngine without importing the full package
spec = importlib.util.spec_from_file_location(
    "menace.quick_fix_engine", os.path.join(os.path.dirname(__file__), "..", "quick_fix_engine.py")
)
quick_fix = importlib.util.module_from_spec(spec)
sys.modules["menace.quick_fix_engine"] = quick_fix
spec.loader.exec_module(quick_fix)
QuickFixEngine = quick_fix.QuickFixEngine


class DummyManager:
    def run_patch(self, path, desc):
        self.calls = getattr(self, "calls", [])
        self.calls.append((path, desc))


class FailingGraph:
    def add_telemetry_event(self, *a, **k):
        raise RuntimeError("boom")

    def update_error_stats(self, *a, **k):
        pass


def test_telemetry_error_logged(monkeypatch, tmp_path, caplog):
    engine = QuickFixEngine(
        error_db=None,
        manager=DummyManager(),
        threshold=1,
        graph=FailingGraph(),
    )
    (tmp_path / "bot.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "bot", {}, 1))
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        engine.run("bot")
    assert "telemetry update failed" in caplog.text


class DummyErrorDB:
    def top_error_module(self, bot):
        return ("runtime_fault", "b", {"a": 1, "b": 2}, 2, bot)


class DummyGraph:
    def __init__(self):
        self.events = []
        self.updated = None

    def add_telemetry_event(self, *a, **k):
        self.events.append((a, k))

    def update_error_stats(self, db):
        self.updated = db


def test_run_targets_frequent_module(tmp_path, monkeypatch):
    engine = QuickFixEngine(
        error_db=DummyErrorDB(),
        manager=DummyManager(),
        threshold=2,
        graph=DummyGraph(),
    )
    (tmp_path / "b.py").write_text("x=1\n")
    monkeypatch.chdir(tmp_path)
    engine.run("bot")
    assert engine.manager.calls[0][0] == Path("b.py")
    assert engine.graph.events[0][0][2] == "b"
    assert engine.graph.updated is engine.db
