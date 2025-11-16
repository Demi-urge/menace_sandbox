import sys, types

class _StubTracker:
    def __init__(self, *a, **k):
        pass
    def origin_db_deltas(self):
        return {}
    def update_db_metrics(self, metrics):
        pass
    def update(self, *a, **k):
        pass

sys.modules.setdefault("roi_tracker", types.SimpleNamespace(ROITracker=_StubTracker))

import cognition_layer as cl_module
from vector_service.cognition_layer import CognitionLayer
from vector_service.patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
import pytest


class DummyEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, payload):  # pragma: no cover - simple stub
        self.events.append((topic, payload))


class DummyContextBuilder:
    def __init__(self, risk_scores, db_weights):
        self.risk_scores = risk_scores
        self.db_weights = dict(db_weights)
        self.retriever = object()

    def refresh_db_weights(self, weights=None, *, vector_metrics=None):
        if weights is None and vector_metrics is not None:
            weights = vector_metrics.get_db_weights()
        if weights:
            self.db_weights.clear()
            self.db_weights.update(weights)

    def build_context(
        self,
        prompt,
        *,
        top_k=5,
        include_vectors=False,
        session_id="",
        return_stats=False,
        return_metadata=False,
    ):
        vectors = [("text", "t1", 0.9), ("code", "c1", 0.8)]
        weighted = sorted(
            vectors,
            key=lambda v: v[2] * self.db_weights.get(v[0], 1.0),
            reverse=True,
        )
        context = " ".join(v[1] for v in weighted[:top_k])
        metadata = {
            "text": [{"vector_id": "t1", "metadata": {"risk_score": self.risk_scores["text"]}}],
            "code": [{"vector_id": "c1", "metadata": {"risk_score": self.risk_scores["code"]}}],
        }
        stats = {"tokens": 0, "wall_time_ms": 0.0, "prompt_tokens": 0}
        sid = session_id or "s"
        return context, sid, vectors, metadata, stats


class DummyROITracker:
    def __init__(self, deltas):
        self.deltas = list(deltas)
        self.metrics = {}

    def origin_db_deltas(self):
        return self.deltas.pop(0) if self.deltas else {}

    def update_db_metrics(self, metrics):
        self.metrics.update({k: dict(v) for k, v in metrics.items()})

    def update(self, *a, **k):  # pragma: no cover - interface stub
        return None


def _setup_layer(tmp_path, monkeypatch, roi_deltas):
    vm = VectorMetricsDB(tmp_path / "vm.db")
    vm.set_db_weights({"text": 0.5, "code": 0.5})
    builder = DummyContextBuilder({"text": 0.0, "code": 0.0}, vm.get_db_weights())
    tracker = DummyROITracker([roi_deltas, roi_deltas])
    bus = DummyEventBus()
    patch_logger = PatchLogger(
        patch_db=object(),
        vector_metrics=vm,
        roi_tracker=tracker,
        event_bus=bus,
        patch_safety=type(
            "PS",
            (),
            {"threshold": 0.0, "load_failures": lambda self, force=False: None, "evaluate": lambda self, *a, **k: (True, 0.0, {})},
        )(),
    )
    layer = CognitionLayer(
        retriever=builder.retriever,
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=vm,
        roi_tracker=tracker,
        event_bus=bus,
    )

    async def fake_backfill(*a, **k):
        return None

    monkeypatch.setattr("vector_service.cognition_layer.schedule_backfill", fake_backfill)
    monkeypatch.setattr(layer, "reload_reliability_scores", lambda: None)

    setattr(builder, "_cognition_layer", layer)
    cl_module._roi_tracker = tracker
    return cl_module, vm, builder


def test_success_updates_metrics_and_weights(tmp_path, monkeypatch):
    cl, vm, builder = _setup_layer(tmp_path, monkeypatch, {"text": 0.2, "code": -0.1})
    ctx, sid = cl.build_cognitive_context("prompt", top_k=2, context_builder=builder)
    assert ctx.split() == ["t1", "c1"]

    cl.log_feedback(sid, True, contribution=1.0, context_builder=builder)

    win_rates = vm.retriever_win_rate_by_db()
    assert win_rates["text"] == pytest.approx(1.0)
    assert win_rates["code"] == pytest.approx(1.0)

    weights = vm.get_db_weights()
    assert weights["text"] > weights["code"]


def test_failure_updates_metrics_and_weights(tmp_path, monkeypatch):
    cl, vm, builder = _setup_layer(tmp_path, monkeypatch, {"text": 0.2, "code": -0.1})
    ctx, sid = cl.build_cognitive_context("prompt", top_k=2, context_builder=builder)
    cl.log_feedback(sid, False, contribution=1.0, context_builder=builder)

    regret_rates = vm.retriever_regret_rate_by_db()
    assert regret_rates["text"] == pytest.approx(1.0)
    assert regret_rates["code"] == pytest.approx(1.0)

    weights = vm.get_db_weights()
    assert weights["code"] > weights["text"]
