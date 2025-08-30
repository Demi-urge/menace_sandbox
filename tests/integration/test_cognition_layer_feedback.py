import pytest
from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def reload_reliability_scores(self):  # pragma: no cover - interface stub
        return None


class DummyROITracker:
    def __init__(self, deltas):
        self.recent_delta_metrics = {
            origin: {"roi": delta} for origin, delta in deltas.items()
        }

    def origin_db_deltas(self):
        return {origin: data["roi"] for origin, data in self.recent_delta_metrics.items()}

    def update_db_metrics(self, metrics):  # pragma: no cover - interface only
        self.last_update = metrics

    def update(self, *_args, **_kwargs):  # pragma: no cover - interface stub
        return None


class DummyPatchLogger:
    def __init__(self, risk_scores):
        self.risk_scores = risk_scores
        self.roi_tracker = None
        self.patch_safety = type(
            "PS",
            (),
            {"threshold": 0.0, "load_failures": lambda self, force=False: None},
        )()

    def track_contributors(
        self,
        vector_ids,
        result,
        *,
        patch_id="",
        session_id="",
        contribution=None,
        retrieval_metadata=None,
        risk_callback=None,
        lines_changed=None,
        tests_passed=None,
        enhancement_name=None,
        start_time=None,
        end_time=None,
        diff=None,
        summary=None,
        outcome=None,
        error_summary=None,
        effort_estimate=None,
    ):
        if risk_callback is not None:
            risk_callback(self.risk_scores)
        return self.risk_scores


class DummyContextBuilder:
    def __init__(self, risk_scores, db_weights=None):
        self.risk_scores = risk_scores
        self.db_weights = dict(db_weights or {})
        self.retriever = DummyRetriever()

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
        return context, session_id or "s", weighted[:top_k], metadata, stats


@pytest.fixture
def roi_deltas():
    return {"text": 0.2, "code": -0.1}


@pytest.fixture
def risk_scores():
    return {"text": 0.4, "code": -0.2}


def test_cognition_layer_feedback(tmp_path, roi_deltas, risk_scores):
    vm = VectorMetricsDB(tmp_path / "vm.db")
    vm.set_db_weights({"text": 0.5, "code": 0.5})

    builder = DummyContextBuilder(risk_scores, vm.get_db_weights())
    logger = DummyPatchLogger(risk_scores)
    tracker = DummyROITracker(roi_deltas)
    layer = CognitionLayer(
        retriever=builder.retriever,
        context_builder=builder,
        patch_logger=logger,
        vector_metrics=vm,
        roi_tracker=tracker,
    )

    context, sid = layer.query("prompt", top_k=2)
    assert context.split() == ["t1", "c1"]

    layer.record_patch_outcome(sid, False)

    weights = vm.get_db_weights()
    assert weights["code"] > weights["text"]

    context2, _ = layer.query("prompt", top_k=2)
    assert context2.split()[0] == "c1"
    assert builder.db_weights["code"] > builder.db_weights["text"]
