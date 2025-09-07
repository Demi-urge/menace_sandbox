import os
import logging
import types
import sys
from dataclasses import dataclass

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "menace",
    Path(__file__).resolve().parents[1] / "__init__.py",  # path-ignore
    submodule_search_locations=[str(Path(__file__).resolve().parents[1])],
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

bus_mod = types.ModuleType("menace.unified_event_bus")
class _Bus:
    def __init__(self, *a, **k):
        pass
    def publish(self, *a, **k):
        pass
bus_mod.UnifiedEventBus = _Bus
class _EventBus:
    pass
bus_mod.EventBus = _EventBus
sys.modules["menace.unified_event_bus"] = bus_mod

kg_mod = types.ModuleType("menace.knowledge_graph")
class _KG:
    def __init__(self, *a, **k):
        pass
    def add_memory_entry(self, *a, **k):
        pass
kg_mod.KnowledgeGraph = _KG
sys.modules["menace.knowledge_graph"] = kg_mod

dpr_mod = types.ModuleType("dynamic_path_router")
dpr_mod.get_project_root = lambda *a, **k: Path(".")
dpr_mod.get_project_roots = lambda *a, **k: [Path(".")]
sys.modules.setdefault("dynamic_path_router", dpr_mod)

from menace.evaluation_history_db import EvaluationHistoryDB
import db_router

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# stub modules required during import
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sk_mod = types.ModuleType("sklearn")
sk_mod.linear_model = types.SimpleNamespace(LogisticRegression=object)
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.linear_model", sk_mod.linear_model)
pred_mod = types.ModuleType("menace.resource_prediction_bot")
@dataclass
class ResourceMetrics:
    cpu: float
    memory: float
    disk: float
    time: float
pred_mod.ResourceMetrics = ResourceMetrics
pred_mod.TemplateDB = object
sys.modules["menace.resource_prediction_bot"] = pred_mod

alloc_mod = types.ModuleType("menace.resource_allocation_bot")
class ResourceAllocationBot:
    def __init__(self, *a, **k):
        pass
    def allocate(self, metrics):
        return [(name, True) for name in metrics]
alloc_mod.ResourceAllocationBot = ResourceAllocationBot
sys.modules["menace.resource_allocation_bot"] = alloc_mod


class _DummyBuilder:
    def build(self, *_: object, **__: object) -> str:
        return "ctx"

    def refresh_db_weights(self):
        pass

stub_ctx = types.ModuleType("vector_service.context_builder")
class ContextBuilder:
    def __init__(self, *a, **k):
        pass
stub_ctx.ContextBuilder = ContextBuilder
sys.modules["vector_service.context_builder"] = stub_ctx

import menace.niche_saturation_bot as ns
from menace.menace_memory_manager import MenaceMemoryManager


def test_saturate_logs_strategy_error(tmp_path, caplog):
    class BadStrategy:
        def receive_niche_info(self, info):
            raise RuntimeError("boom")

    alloc = ResourceAllocationBot(context_builder=_DummyBuilder())
    bot = ns.NicheSaturationBot(
        db=ns.NicheDB(tmp_path / "n.db"),
        alloc_bot=alloc,
        strategy_bot=BadStrategy(),
        context_builder=_DummyBuilder(),
    )
    caplog.set_level(logging.ERROR)
    bot.saturate([ns.NicheCandidate("x", 1.0, 0.0)])
    assert "strategy bot failed" in caplog.text


def test_summary_logs_store_error(tmp_path, caplog):
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm.store("k", "one. two. three.")

    def oops(*args, **kwargs):
        raise RuntimeError("boom")

    mm.store = oops
    caplog.set_level(logging.ERROR)
    mm.summarise_memory("k")
    assert "Failed to store summary" in caplog.text


def test_ensure_config_logs_error(monkeypatch, caplog, tmp_path):
    import menace.config_discovery as cd

    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    caplog.set_level(logging.ERROR)
    cd.ensure_config(["X"], save_path=tmp_path / "cfg")
    assert "Failed to save config" in caplog.text


def test_evaluation_failure_logged(tmp_path, caplog):
    import types, sys
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_mod.Engine = object
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)
    sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
    from menace.evaluation_manager import EvaluationManager

    class BadEngine:
        def evaluate(self):
            raise RuntimeError("boom")

    router = db_router.DBRouter(
        "log", str(tmp_path / "hist.db"), str(tmp_path / "hist.db")
    )
    db = EvaluationHistoryDB(router=router)
    mgr = EvaluationManager(learning_engine=BadEngine(), history_db=db)
    caplog.set_level(logging.ERROR)
    mgr.evaluate_all()
    assert "evaluation failed for learning_engine" in caplog.text
    hist = db.history("learning_engine")
    assert hist and hist[0][2] == 0 and "boom" in hist[0][3]
