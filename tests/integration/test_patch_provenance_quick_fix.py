import os
import sys
import types
import importlib.util
from pathlib import Path

# Create lightweight package stubs to load QuickFixEngine without heavy deps
package = types.ModuleType("menace")
sys.modules["menace"] = package

error_bot = types.ModuleType("menace.error_bot")
error_bot.ErrorDB = object
sys.modules["menace.error_bot"] = error_bot

scm = types.ModuleType("menace.self_coding_manager")
scm.SelfCodingManager = object
sys.modules["menace.self_coding_manager"] = scm

kg = types.ModuleType("menace.knowledge_graph")
kg.KnowledgeGraph = object
sys.modules["menace.knowledge_graph"] = kg

# Load QuickFixEngine
spec = importlib.util.spec_from_file_location(
    "menace.quick_fix_engine",
    Path(__file__).resolve().parents[2] / "quick_fix_engine.py",  # path-ignore
)
quick_fix = importlib.util.module_from_spec(spec)
sys.modules["menace.quick_fix_engine"] = quick_fix
spec.loader.exec_module(quick_fix)
QuickFixEngine = quick_fix.QuickFixEngine


class DummyGraph:
    def add_telemetry_event(self, *a, **k):
        pass

    def update_error_stats(self, *a, **k):
        pass


def test_quick_fix_records_license_and_alerts(tmp_path, monkeypatch):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
    from code_database import PatchHistoryDB, PatchRecord
    from patch_provenance import get_patch_provenance
    from vector_service.patch_logger import PatchLogger

    db = PatchHistoryDB()
    patch_id = db.add(PatchRecord("mod.py", "desc", 1.0, 2.0))  # path-ignore

    class Manager:
        def __init__(self):
            self.bot_registry = types.SimpleNamespace()
            self.data_bot = types.SimpleNamespace()

        def run_patch(self, path, desc, context_meta=None, **kw):
            return types.SimpleNamespace(patch_id=patch_id)

        def register_bot(self, *a, **k):
            return None

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

    class DummyBuilder:
        def refresh_db_weights(self):
            return None

        def build(self, query, session_id=None, include_vectors=False):
            return ""

    engine = QuickFixEngine(
        error_db=None,
        manager=Manager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=Retriever(),
        patch_logger=PatchLogger(patch_db=db),
        context_builder=DummyBuilder(),
        helper_fn=lambda *a, **k: "",
    )
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)

    engine.run("bot")
    prov = get_patch_provenance(patch_id, patch_db=db)
    assert prov[0]["license"] == "mit"
    assert prov[0]["semantic_alerts"] == ["unsafe"]
