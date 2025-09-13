import os
import sys
import types
import importlib.util
from pathlib import Path

from vector_service import FallbackResult

ROOT = Path(__file__).resolve().parents[2]

# Ensure menace package with required stubs exists
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", pkg)

# Stub minimal modules required by quick_fix_engine and error_cluster_predictor
stub_modules = {
    "error_bot": ["ErrorDB"],
    "self_coding_manager": ["SelfCodingManager"],
    "knowledge_graph": ["KnowledgeGraph", "_SimpleKMeans"],
    "human_alignment_flagger": ["_collect_diff_data"],
    "human_alignment_agent": ["HumanAlignmentAgent"],
    "violation_logger": ["log_violation"],
    "vector_metrics_db": ["VectorMetricsDB"],
}
for name, attrs in stub_modules.items():
    mod = types.ModuleType(f"menace.{name}")
    for attr in attrs:
        if attr == "_collect_diff_data":
            setattr(mod, attr, lambda *a, **k: {})
        elif attr == "log_violation":
            setattr(mod, attr, lambda *a, **k: None)
        elif attr == "_SimpleKMeans":
            class _SimpleKMeans:
                def __init__(self, n_clusters=1):
                    self.n_clusters = n_clusters
                def fit(self, vectors):
                    return None
                def predict(self, vectors):
                    return [0 for _ in vectors]
            setattr(mod, attr, _SimpleKMeans)
        else:
            setattr(mod, attr, type(attr, (), {}))
    sys.modules[f"menace.{name}"] = mod

# Stub for modules imported without package prefix
api_stub = types.ModuleType("adaptive_roi_predictor")
api_stub.load_training_data = lambda *a, **k: []
sys.modules.setdefault("adaptive_roi_predictor", api_stub)


# Load modules using package-style paths so relative imports resolve

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

error_cluster_predictor = _load("menace.error_cluster_predictor", "error_cluster_predictor.py")  # path-ignore
ErrorClusterPredictor = error_cluster_predictor.ErrorClusterPredictor

quick_fix = _load("menace.quick_fix_engine", "quick_fix_engine.py")  # path-ignore
QuickFixEngine = quick_fix.QuickFixEngine

import sandbox_runner.cycle as cycle


def test_sandbox_runner_uses_vector_service():
    assert cycle.Retriever.__module__.startswith("vector_service")
    assert cycle.PatchLogger.__module__.startswith("vector_service")


def test_quick_fix_engine_uses_vector_service_retriever():
    class SpyRetriever:
        def __init__(self):
            self.calls = []
        def search(self, query, top_k, session_id):
            self.calls.append((query, top_k, session_id))
            return FallbackResult("no results", [], 0.0)
    class DummyBuilder:
        def refresh_db_weights(self):
            return None

        def build(self, query, session_id=None, include_vectors=False):
            return ""
    manager = types.SimpleNamespace(
        bot_registry=object(), data_bot=object(), register_bot=lambda *a, **k: None
    )
    engine = QuickFixEngine(
        error_db=object(),
        manager=manager,
        retriever=SpyRetriever(),
        context_builder=DummyBuilder(),
    )
    hits, sid, vectors = engine._redundant_retrieve("m", 1)
    assert hits == [] and sid == "" and vectors == []
    assert engine.retriever.calls


def test_error_cluster_predictor_uses_vector_service_retriever(monkeypatch):
    class SpyRetriever:
        def __init__(self):
            self.calls = []
        def search(self, query, top_k, session_id):
            self.calls.append(query)
            return FallbackResult("no results", [], 0.0)
    predictor = ErrorClusterPredictor(graph=object(), db=object(), retriever=SpyRetriever())
    monkeypatch.setattr(predictor, "_module_vectors", lambda: (["module:a"], [[1.0]]))
    predictor.predict_high_risk_modules()
    assert predictor.retriever.calls == ["a"]
