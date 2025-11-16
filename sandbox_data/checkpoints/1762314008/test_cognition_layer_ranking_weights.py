import pytest
from unittest.mock import MagicMock

from vector_metrics_db import VectorMetricsDB
import menace_sandbox.roi_tracker as rt
from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer
from vector_service.patch_logger import PatchLogger
from patch_safety import PatchSafety


class DummyRetriever:
    def __init__(self):
        self.hits = [
            {
                "origin_db": "bot",
                "record_id": "b1",
                "score": 0.3,
                "text": "bot one",
                "metadata": {"name": "bot-one", "redacted": True},
            },
            {
                "origin_db": "bot",
                "record_id": "b2",
                "score": 0.2,
                "text": "bot two",
                "metadata": {"name": "bot-two", "redacted": True},
            },
            {
                "origin_db": "workflow",
                "record_id": "w1",
                "score": 0.4,
                "text": "workflow",
                "metadata": {"title": "wf", "redacted": True},
            },
            {
                "origin_db": "enhancement",
                "record_id": "e1",
                "score": 0.1,
                "text": "enh",
                "metadata": {"title": "enh", "redacted": True},
            },
            {
                "origin_db": "error",
                "record_id": "er1",
                "score": 0.05,
                "metadata": {"message": "err", "redacted": True},
            },
        ]

    def search(self, query: str, top_k: int = 5, session_id: str = "", **_):
        return list(self.hits)


class DummyBus:
    def publish(self, *_, **__):
        pass


def _make_layer(tmp_path, with_tracker=True):
    retriever = DummyRetriever()
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    tracker = rt.ROITracker() if with_tracker else None
    builder = ContextBuilder(retriever=retriever, roi_tracker=tracker)
    patch_logger = PatchLogger(
        vector_metrics=vec_db,
        roi_tracker=tracker,
        event_bus=DummyBus(),
        patch_safety=PatchSafety(failure_db_path=None),
    )
    layer = CognitionLayer(
        retriever=retriever,
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=vec_db,
        roi_tracker=tracker,
        event_bus=patch_logger.event_bus,
    )
    return layer, vec_db, tracker


def test_patch_success_updates_metrics_and_weights(tmp_path):
    layer, vec_db, tracker = _make_layer(tmp_path)
    _ctx, sid = layer.query("hello", top_k=5)
    baseline = {}
    for origin, _vid, score in layer._session_vectors[sid]:
        baseline[origin] = max(score, baseline.get(origin, float("-inf")))
    assert baseline["workflow"] > baseline["bot"]
    layer.record_patch_outcome(sid, True, contribution=2.0)
    weights = vec_db.get_db_weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    for origin in ("bot", "workflow", "enhancement", "error"):
        assert weights[origin] == pytest.approx(0.25)
        assert tracker.origin_db_delta_history[origin][-1] == pytest.approx(2.0)
        assert vec_db.retriever_win_rate(origin) == pytest.approx(1.0)
    ctx2, _sid2, vectors2 = layer.context_builder.build_context(
        "hello", top_k=5, include_vectors=True
    )
    assert {o for o, _vid, _s in vectors2} == {
        "bot",
        "workflow",
        "enhancement",
        "error",
    }


def test_patch_failure_decreases_weights(tmp_path):
    layer, vec_db, tracker = _make_layer(tmp_path)
    _ctx, sid = layer.query("hello", top_k=5)
    layer.record_patch_outcome(sid, False, contribution=-1.0)
    weights = vec_db.get_db_weights()
    assert weights["bot"] == pytest.approx(0.0)
    assert weights["workflow"] == pytest.approx(0.0)
    assert tracker.origin_db_delta_history["bot"][-1] == pytest.approx(-1.0)
    assert vec_db.retriever_regret_rate("bot") == pytest.approx(1.0)


def test_high_risk_vectors_penalize_weights(tmp_path, monkeypatch):
    layer, vec_db, _ = _make_layer(tmp_path, with_tracker=False)
    _ctx, sid = layer.query("hello", top_k=5)
    # simulate PatchLogger reporting a risk score for bot vectors
    monkeypatch.setattr(
        layer.patch_logger, "track_contributors", lambda *a, **k: {"bot": 2.0}
    )
    layer.record_patch_outcome(sid, True, contribution=1.0)
    weights = vec_db.get_db_weights()
    assert weights["bot"] == pytest.approx(0.0)
    assert weights["workflow"] > weights["bot"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_update_ranker_applies_roi_and_risk_penalties(tmp_path, monkeypatch):
    layer, vec_db, _ = _make_layer(tmp_path, with_tracker=False)
    vectors = [
        ("bot", "b1", 0.0),
        ("workflow", "w1", 0.0),
        ("error", "er1", 0.0),
    ]
    roi_deltas = {"bot": 1.0, "workflow": 0.5, "error": 1.5}
    risk_scores = {"bot": 0.2, "workflow": 0.6, "error": 2.0}

    spy = MagicMock(side_effect=vec_db.update_db_weight)
    monkeypatch.setattr(vec_db, "update_db_weight", spy)

    updates = layer.update_ranker(
        vectors, True, roi_deltas=roi_deltas, risk_scores=risk_scores
    )

    expected_calls = {"bot": 1.0, "workflow": 0.5, "error": -0.5}
    calls = {c.args[0]: c.args[1] for c in spy.call_args_list}
    for origin, delta in expected_calls.items():
        assert calls[origin] == pytest.approx(delta)

    expected_weights = {"bot": pytest.approx(2 / 3), "workflow": pytest.approx(1 / 3), "error": 0.0}
    for origin, wt in expected_weights.items():
        assert updates[origin] == wt

    weights = vec_db.get_db_weights()
    for origin, wt in expected_weights.items():
        assert weights[origin] == pytest.approx(wt)
        assert layer.context_builder.db_weights[origin] == pytest.approx(wt)

    assert weights["error"] == pytest.approx(0.0)


def test_update_ranker_logs_weight_deltas(tmp_path):
    layer, vec_db, _ = _make_layer(tmp_path, with_tracker=False)
    vectors = [("A", "v1", 0.0)]
    layer.update_ranker(vectors, True, roi_deltas={"A": 0.5})
    rows = vec_db.conn.execute(
        "SELECT db, contribution, similarity FROM vector_metrics WHERE event_type='ranker'"
    ).fetchall()
    assert rows == [("A", 0.5, 0.5)]
