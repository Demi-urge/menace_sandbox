from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


LEGACY_FAILURE_SIGNATURE = "attempted relative import with no known parent package"


def _install_top_level_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_error_cluster_predictor_shim_imports_without_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_top_level_dependency_stubs(monkeypatch)

    shim_module_name = "menace.error_cluster_predictor"
    top_level_module_name = "error_cluster_predictor"

    monkeypatch.delitem(sys.modules, shim_module_name, raising=False)
    monkeypatch.delitem(sys.modules, top_level_module_name, raising=False)

    real_import_module = importlib.import_module

    def _import_module_with_missing_impl(name: str, package: str | None = None):
        if name == "menace.error_cluster_predictor_impl":
            raise ModuleNotFoundError(
                "No module named 'menace.error_cluster_predictor_impl'",
                name="menace.error_cluster_predictor_impl",
            )
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module_with_missing_impl)

    try:
        module = importlib.import_module(shim_module_name)
    except Exception as exc:  # pragma: no cover - assertion-focused failure path
        assert LEGACY_FAILURE_SIGNATURE not in str(exc)
        pytest.fail(f"shim import unexpectedly failed: {exc!r}")

    assert hasattr(module, "ErrorClusterPredictor")
