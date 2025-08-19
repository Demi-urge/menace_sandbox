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

# Stubs for modules imported by patch_provenance/code_database
def _auto_link(*a, **k):
    def decorator(func):
        return func
    return decorator

sys.modules.setdefault("auto_link", types.SimpleNamespace(auto_link=_auto_link))
sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "retry_utils", types.SimpleNamespace(publish_with_retry=lambda *a, **k: None, with_retry=lambda *a, **k: None)
)
sys.modules.setdefault(
    "alert_dispatcher", types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={})
)

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
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
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
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    engine.run("bot")
    assert engine.manager.calls[0][0] == Path("b.py")
    assert engine.graph.events[0][0][2] == "b"
    assert engine.graph.events[0][1]["resolved"] is True
    assert engine.graph.updated is engine.db


class DummyPreemptiveDB:
    def __init__(self):
        self.records = []

    def log_preemptive_patch(self, module, risk, patch_id):
        self.records.append((module, risk, patch_id))


class DummyResult:
    def __init__(self, patch_id):
        self.patch_id = patch_id


class DummyManager2:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def run_patch(self, path, desc):
        self.calls.append((path, desc))
        if self.fail:
            raise RuntimeError("boom")
        return DummyResult(123)


def test_preemptive_patch_modules(tmp_path, monkeypatch):
    db = DummyPreemptiveDB()
    mgr = DummyManager2()
    engine = QuickFixEngine(error_db=db, manager=mgr, threshold=0, graph=None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("x=1\n")
    modules = [("mod", 0.9), ("low", 0.1)]
    engine.preemptive_patch_modules(modules, risk_threshold=0.5)
    assert mgr.calls == [(Path("mod.py"), "preemptive_patch")]
    assert db.records == [("mod", 0.9, 123)]


def test_preemptive_patch_falls_back(monkeypatch, tmp_path):
    db = DummyPreemptiveDB()
    mgr = DummyManager2(fail=True)
    engine = QuickFixEngine(error_db=db, manager=mgr, threshold=0, graph=None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("x=1\n")
    monkeypatch.setattr(quick_fix, "generate_patch", lambda m, engine=None: 999)
    engine.preemptive_patch_modules([("mod", 0.8)], risk_threshold=0.5)
    assert mgr.calls == [(Path("mod.py"), "preemptive_patch")]
    assert db.records == [("mod", 0.8, 999)]


def test_generate_patch_blocks_risky(monkeypatch, tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x=1\n")

    class DummyEngine:
        def apply_patch(self, p, *a, **k):
            with open(p, "a", encoding="utf-8") as f:
                f.write("eval('2')\n")
            return 1, "", ""

    monkeypatch.chdir(tmp_path)
    res = quick_fix.generate_patch(str(path), engine=DummyEngine())
    assert res is None
    assert path.read_text() == "x=1\n"


def test_run_records_retrieval_metadata(tmp_path, monkeypatch):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
    from code_database import PatchHistoryDB, PatchRecord
    from patch_provenance import get_patch_provenance
    from vector_service.patch_logger import PatchLogger

    db = PatchHistoryDB()
    patch_id = db.add(PatchRecord("mod.py", "desc", 1.0, 2.0))

    class Manager:
        def run_patch(self, path, desc, context_meta=None):
            return types.SimpleNamespace(patch_id=patch_id)

    class Retriever:
        def search(self, query, top_k, session_id):
            return [
                {
                    "origin_db": "db1",
                    "record_id": "vec1",
                    "score": 0.5,
                    "license": "mit",
                    "semantic_alerts": ["unsafe"],
                }
            ]

    engine = QuickFixEngine(
        error_db=None,
        manager=Manager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=Retriever(),
        patch_logger=PatchLogger(patch_db=db),
    )
    (tmp_path / "mod.py").write_text("x=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    engine.run("bot")
    prov = get_patch_provenance(patch_id, patch_db=db)
    assert prov[0]["license"] == "mit"
    assert prov[0]["semantic_alerts"] == ["unsafe"]


def test_run_records_ancestry_without_logger(tmp_path, monkeypatch):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
    from code_database import PatchHistoryDB, PatchRecord
    from patch_provenance import get_patch_provenance

    db = PatchHistoryDB()
    patch_id = db.add(PatchRecord("mod.py", "desc", 1.0, 2.0))

    class Manager:
        def __init__(self):
            self.engine = types.SimpleNamespace(patch_db=db)

        def run_patch(self, path, desc, context_meta=None):
            return types.SimpleNamespace(patch_id=patch_id)

    class Retriever:
        def search(self, query, top_k, session_id):
            return [{"origin_db": "db1", "record_id": "vec1", "score": 0.4}]

    engine = QuickFixEngine(
        error_db=None,
        manager=Manager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=Retriever(),
    )
    (tmp_path / "mod.py").write_text("x=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)

    engine.run("bot")
    prov = get_patch_provenance(patch_id, patch_db=db)
    assert prov[0]["vector_id"] == "vec1"
