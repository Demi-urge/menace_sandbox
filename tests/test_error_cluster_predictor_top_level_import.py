from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def test_error_cluster_predictor_imports_top_level_without_package_context(monkeypatch):
    module_name = "error_cluster_predictor"
    module_path = Path(__file__).resolve().parents[1] / "error_cluster_predictor.py"

    knowledge_graph_stub = ModuleType("knowledge_graph")

    class KnowledgeGraph:  # noqa: D401 - test stub
        pass

    class _SimpleKMeans:  # noqa: D401 - test stub
        def __init__(self, *args, **kwargs):
            pass

    knowledge_graph_stub.KnowledgeGraph = KnowledgeGraph
    knowledge_graph_stub._SimpleKMeans = _SimpleKMeans

    vector_service_stub = ModuleType("vector_service")

    class Retriever:  # noqa: D401 - test stub
        pass

    class FallbackResult:  # noqa: D401 - test stub
        pass

    class ErrorResult(Exception):
        pass

    vector_service_stub.Retriever = Retriever
    vector_service_stub.FallbackResult = FallbackResult
    vector_service_stub.ErrorResult = ErrorResult

    monkeypatch.setitem(sys.modules, "knowledge_graph", knowledge_graph_stub)
    monkeypatch.setitem(sys.modules, "vector_service", vector_service_stub)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ErrorClusterPredictor is not None
