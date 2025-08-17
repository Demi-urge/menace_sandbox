import logging
from types import SimpleNamespace
import importlib.util
import os
import sys
import types

from semantic_service import Retriever, FallbackResult

# Construct minimal menace package for QuickFixEngine
package = types.ModuleType("menace")
package.__path__ = [os.path.dirname(__file__)]
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

spec = importlib.util.spec_from_file_location(
    "menace.quick_fix_engine", os.path.join(os.path.dirname(__file__), "..", "quick_fix_engine.py")
)
qfe = importlib.util.module_from_spec(spec)
sys.modules["menace.quick_fix_engine"] = qfe
spec.loader.exec_module(qfe)
class DummyUR:
    def __init__(self, score=0.5):
        self.score = score
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        hit = SimpleNamespace(origin_db="bot", record_id="1", score=self.score, metadata={}, reason="")
        return [hit], "sid", [("bot", "1")]


def test_logging_hook_emits_session_id(caplog):
    ur = DummyUR()
    retriever = Retriever(retriever=ur)
    caplog.set_level(logging.INFO)
    retriever.search("alpha", session_id="sess-1")
    assert ur.calls == [("alpha", 5)]
    assert any("Retriever.search" in r.message and r.session_id == "sess-1" for r in caplog.records)


def test_retriever_fallback_low_similarity():
    ur = DummyUR(score=0.05)
    retriever = Retriever(retriever=ur)
    hits = retriever.search("beta")
    assert isinstance(hits, FallbackResult)
    assert hits.reason == "low confidence"
    assert list(hits)[0]["origin_db"] == "heuristic"


class DummyManager:
    def run_patch(self, path, desc, context_meta=None):  # pragma: no cover - simple stub
        return SimpleNamespace(patch_id=1)


class DummyGraph:
    def add_telemetry_event(self, *a, **k):
        pass

    def update_error_stats(self, db):
        pass


class DummyPatchLogger:
    def __init__(self):
        self.calls = []

    def track_contributors(self, ids, result, *, patch_id="", session_id=""):
        self.calls.append((ids, result, patch_id, session_id))


def test_quick_fix_engine_uses_service_layer(monkeypatch, tmp_path):
    retr = Retriever(retriever=DummyUR())
    plog = DummyPatchLogger()
    engine = qfe.QuickFixEngine(
        error_db=SimpleNamespace(),
        manager=DummyManager(),
        graph=DummyGraph(),
        threshold=1,
        retriever=retr,
        patch_logger=plog,
    )
    (tmp_path / "mod.py").write_text("x=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qfe, "_VEC_METRICS", None)
    monkeypatch.setattr(qfe.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    engine.run("bot")
    assert retr.retriever.calls  # underlying retriever was invoked
    assert plog.calls  # patch logger recorded contributors
