import importlib.util
import sys
import types
from pathlib import Path
import pytest

pytest.importorskip("sqlalchemy")

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod
ur = _load("universal_retriever")
sys.modules["universal_retriever"] = ur
dr = _load("database_router")
DatabaseRouter = dr.DatabaseRouter

def test_router_queries_highest_reliability_first(monkeypatch):
    router = DatabaseRouter()
    calls = []

    def fake_retrieve(query, top_k=10, dbs=None):
        calls.append(dbs)
        return [], "", []

    monkeypatch.setattr(router._retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        router._retriever,
        "reliability_metrics",
        lambda: {"information": {"reliability": 0.9}, "bot": {"reliability": 0.2}},
    )
    router.min_reliability = 0.5
    router.redundancy_limit = 2
    router.semantic_search("query")
    assert calls == [["information"]]

def test_router_fallback_on_low_reliability(monkeypatch):
    router = DatabaseRouter()
    calls = []

    def fake_retrieve(query, top_k=10, dbs=None):
        calls.append(dbs)
        return [], "", []

    monkeypatch.setattr(router._retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        router._retriever,
        "reliability_metrics",
        lambda: {
            "information": {"reliability": 0.2},
            "bot": {"reliability": 0.4},
            "workflow": {"reliability": 0.3},
        },
    )
    router.min_reliability = 0.5
    router.redundancy_limit = 2
    router.semantic_search("query")
    assert calls == [["bot"], ["workflow"]]
