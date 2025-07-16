import os
import sys
import types
import importlib.util
import logging
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
        pass


class FailingGraph:
    def add_telemetry_event(self, *a, **k):
        raise RuntimeError("boom")


def test_telemetry_error_logged(monkeypatch, tmp_path, caplog):
    engine = QuickFixEngine(
        error_db=None,
        manager=DummyManager(),
        threshold=1,
        graph=FailingGraph(),
    )
    (tmp_path / "bot.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", 1))
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        engine.run("bot")
    assert "telemetry update failed" in caplog.text
