from types import SimpleNamespace

import universal_retriever as ur_mod
from vector_service.cognition_layer import CognitionLayer
from vector_service.retriever import Retriever


def test_reliability_reload_calls_universal_retriever(monkeypatch):
    calls = []

    # Avoid any database interactions during UniversalRetriever init
    monkeypatch.setattr(
        ur_mod.UniversalRetriever, "_load_reliability_stats", lambda self: None
    )

    # Stub the reload method to record calls
    def record_call(self) -> None:
        calls.append(True)

    monkeypatch.setattr(
        ur_mod.UniversalRetriever, "reload_reliability_scores", record_call
    )

    # Minimal instance that satisfies constructor requirements
    ur = ur_mod.UniversalRetriever(
        enable_model_ranking=False,
        enable_reliability_bias=False,
        code_db=object(),
    )

    builder = SimpleNamespace()
    patch_logger = SimpleNamespace(roi_tracker=None, event_bus=None)
    layer = CognitionLayer(
        context_builder=builder,
        retriever=Retriever(retriever=ur),
        patch_logger=patch_logger,
        vector_metrics=None,
        roi_tracker=None,
    )

    layer.reload_reliability_scores()

    assert calls, "UniversalRetriever.reload_reliability_scores was not called"
