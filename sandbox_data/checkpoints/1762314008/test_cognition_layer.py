import pytest
import time

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB
from vector_service.patch_logger import PatchLogger
from patch_safety import PatchSafety


class DummyRetriever:
    def __init__(self, results=None):
        # results is list of (origin, vector_id, score)
        self.results = results or []
        self.calls = []

    def search(self, query, top_k=5, session_id=""):
        self.calls.append((query, top_k, session_id))
        return list(self.results)


class DummyRankingModel:
    def __init__(self):
        self.rank_calls = 0

    def rank(self, items):
        self.rank_calls += 1
        # items is list of (origin, vector_id, score)
        return sorted(items, key=lambda t: t[2], reverse=True)


class DummyContextBuilder:
    def __init__(self, retriever, ranking_model=None):
        self.retriever = retriever
        self.ranking_model = ranking_model

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
        vectors = self.retriever.search(prompt, top_k=top_k, session_id=session_id)
        if self.ranking_model is not None:
            vectors = self.ranking_model.rank(vectors)
        vectors = vectors[:top_k]
        stats = {
            "tokens": len(prompt.split()),
            "wall_time_ms": 1.0,
            "prompt_tokens": len(prompt.split()),
        }
        sid = session_id or "sid"
        meta = {"misc": []}
        ts = time.time() - 30.0
        for origin, vec_id, _ in vectors:
            meta["misc"].append(
                {
                    "origin_db": origin,
                    "vector_id": vec_id,
                    "metadata": {"timestamp": ts},
                }
            )
        if include_vectors:
            if return_metadata:
                if return_stats:
                    return "ctx", sid, vectors, meta, stats
                return "ctx", sid, vectors, meta
            if return_stats:
                return "ctx", sid, vectors, stats
            return "ctx", sid, vectors
        if return_metadata:
            if return_stats:
                return "ctx", meta, stats
            return "ctx", meta
        if return_stats:
            return "ctx", stats
        return "ctx"


class DummyROITracker:
    def __init__(self):
        self.db_metrics = []
        self.updates = []

    def update_db_metrics(self, metrics):
        self.db_metrics.append(metrics)

    def update(self, roi_before, roi_after, *, retrieval_metrics=None, **_):
        self.updates.append(
            {
                "roi_before": roi_before,
                "roi_after": roi_after,
                "retrieval_metrics": retrieval_metrics or [],
            }
        )


class DummyPatchLogger:
    def __init__(self, vector_metrics, roi_tracker):
        self.vector_metrics = vector_metrics
        self.roi_tracker = roi_tracker
        self.calls = []

    def track_contributors(
        self,
        vector_ids,
        result,
        *,
        patch_id="",
        session_id="",
        contribution=None,
        retrieval_metadata=None,
        lines_changed=None,
        tests_passed=None,
        enhancement_name=None,
        start_time=None,
        end_time=None,
        diff=None,
        summary=None,
        outcome=None,
        error_summary=None,
    ):
        self.calls.append(
            {
                "vector_ids": vector_ids,
                "result": result,
                "patch_id": patch_id,
                "session_id": session_id,
                "contribution": contribution,
                "retrieval_metadata": retrieval_metadata,
                "lines_changed": lines_changed,
                "tests_passed": tests_passed,
                "enhancement_name": enhancement_name,
                "end_time": end_time,
                "diff": diff,
                "summary": summary,
                "outcome": outcome,
            }
        )
        origin_totals = {}
        roi_base = 0.0 if contribution is None else contribution
        for vid, score in vector_ids:
            if ":" in vid:
                origin, vec_id = vid.split(":", 1)
            else:
                origin, vec_id = "", vid
            roi = roi_base if contribution is not None else score
            origin_totals[origin] = origin_totals.get(origin, 0.0) + roi
            try:
                self.vector_metrics.update_outcome(
                    session_id,
                    [(origin, vec_id)],
                    contribution=roi,
                    patch_id=patch_id,
                    win=result,
                    regret=not result,
                )
            except Exception:
                pass
        for origin, roi in origin_totals.items():
            try:
                self.vector_metrics.log_retrieval_feedback(
                    origin, win=result, regret=not result, roi=roi
                )
            except Exception:
                pass
        if origin_totals and self.roi_tracker is not None:
            metrics = {
                origin: {
                    "roi": roi,
                    "win_rate": 1.0 if result else 0.0,
                    "regret_rate": 0.0 if result else 1.0,
                }
                for origin, roi in origin_totals.items()
            }
            try:
                self.roi_tracker.update_db_metrics(metrics)
            except Exception:
                pass
        return {}


def _make_layer(results):
    retriever = DummyRetriever(results)
    ranker = DummyRankingModel()
    tracker = DummyROITracker()
    metrics = VectorMetricsDB(":memory:")
    builder = DummyContextBuilder(retriever, ranking_model=ranker)
    logger = DummyPatchLogger(metrics, tracker)
    layer = CognitionLayer(
        context_builder=builder,
        patch_logger=logger,
        vector_metrics=metrics,
        roi_tracker=tracker,
    )
    return layer, retriever, ranker, tracker, metrics, logger


def _make_layer_with_patch_safety(results):
    retriever = DummyRetriever(results)
    ranker = DummyRankingModel()
    tracker = DummyROITracker()
    metrics = VectorMetricsDB(":memory:")
    builder = DummyContextBuilder(retriever, ranking_model=ranker)
    ps = PatchSafety(failure_db_path=None)
    logger = PatchLogger(vector_metrics=metrics, roi_tracker=tracker, patch_safety=ps)
    layer = CognitionLayer(
        context_builder=builder,
        patch_logger=logger,
        vector_metrics=metrics,
        roi_tracker=tracker,
    )
    return layer, retriever, ranker, tracker, metrics, logger


def test_query_and_record_patch_outcome_updates_metrics_and_ranking():
    results = [("db2", "v2", 0.2), ("db1", "v1", 0.9)]
    layer, retriever, ranker, tracker, metrics, logger = _make_layer(results)

    ctx, sid = layer.query("hello world")
    assert ctx == "ctx"
    assert ranker.rank_calls == 1

    rows = metrics.conn.execute(
        "SELECT session_id, vector_id, contribution, win, regret, tokens, wall_time_ms, "
        "prompt_tokens, age FROM vector_metrics WHERE event_type='retrieval' AND session_id=?",
        (sid,),
    ).fetchall()
    assert len(rows) >= 2
    assert all(row[0] == sid for row in rows)
    assert all(row[2] == 0.0 and row[3] is None and row[4] is None for row in rows)
    for _sid, _vid, _c, _w, _r, tokens, wall_ms, prompt_tokens, age in rows:
        assert tokens > 0
        assert wall_ms > 0.0
        assert prompt_tokens > 0
        assert age == pytest.approx(30.0, abs=2.0)

    layer.record_patch_outcome(
        sid,
        True,
        contribution=1.0,
        lines_changed=3,
        tests_passed=True,
        enhancement_name="improve",
        timestamp=123.0,
    )

    assert len(logger.calls) == 1
    assert logger.calls[0]["session_id"] == sid
    assert logger.calls[0]["vector_ids"][0][0] == "db1:v1"
    assert logger.calls[0]["lines_changed"] == 3
    assert logger.calls[0]["tests_passed"] is True
    assert logger.calls[0]["enhancement_name"] == "improve"
    assert logger.calls[0]["timestamp"] == 123.0

    rows = metrics.conn.execute(
        "SELECT contribution, win, regret FROM vector_metrics WHERE session_id=?",
        (sid,),
    ).fetchall()
    assert all(row[0] == pytest.approx(1.0) for row in rows)
    assert all(row[1] == 1 and row[2] == 0 for row in rows)

    assert tracker.db_metrics[0] == {
        "db1": {"roi": 1.0, "win_rate": 1.0, "regret_rate": 0.0},
        "db2": {"roi": 1.0, "win_rate": 1.0, "regret_rate": 0.0},
    }
    assert tracker.updates[0]["roi_after"] == pytest.approx(2.0)
    assert len(tracker.updates[0]["retrieval_metrics"]) == 2


def test_record_patch_outcome_missing_session():
    layer, retriever, ranker, tracker, metrics, logger = _make_layer([])
    layer.record_patch_outcome("missing", True)
    assert logger.calls == []
    assert tracker.db_metrics == []


def test_query_with_no_vectors():
    layer, retriever, ranker, tracker, metrics, logger = _make_layer([])
    _, sid = layer.query("nothing here")
    layer.record_patch_outcome(sid, True)
    assert logger.calls == []
    rows = metrics.conn.execute(
        "SELECT 1 FROM vector_metrics WHERE event_type='retrieval'"
    ).fetchall()
    assert rows == []


def test_session_persistence_and_cleanup(tmp_path):
    db_file = tmp_path / "metrics.db"
    metrics1 = VectorMetricsDB(db_file)
    retriever = DummyRetriever([("db1", "v1", 0.5)])
    ranker = DummyRankingModel()
    tracker = DummyROITracker()
    builder = DummyContextBuilder(retriever, ranking_model=ranker)
    logger1 = DummyPatchLogger(metrics1, tracker)
    layer1 = CognitionLayer(
        context_builder=builder,
        patch_logger=logger1,
        vector_metrics=metrics1,
        roi_tracker=tracker,
    )
    _ctx, sid = layer1.query("hello")
    rows = metrics1.conn.execute(
        "SELECT session_id FROM pending_sessions",
    ).fetchall()
    assert rows == [(sid,)]
    metrics1.conn.close()

    metrics2 = VectorMetricsDB(db_file)
    tracker2 = DummyROITracker()
    logger2 = DummyPatchLogger(metrics2, tracker2)
    builder2 = DummyContextBuilder(retriever, ranking_model=ranker)
    layer2 = CognitionLayer(
        context_builder=builder2,
        patch_logger=logger2,
        vector_metrics=metrics2,
        roi_tracker=tracker2,
    )
    assert sid in layer2._session_vectors
    layer2.record_patch_outcome(sid, True, contribution=1.0)
    rows = metrics2.conn.execute(
        "SELECT session_id FROM pending_sessions",
    ).fetchall()
    assert rows == []


def test_failed_sessions_record_failure():
    results = [("error", "1", 0.9)]
    layer, retriever, ranker, tracker, metrics, logger = _make_layer_with_patch_safety(results)
    ctx, sid = layer.query("boom")
    key = "error:1"
    layer._retrieval_meta[sid][key] = {"category": "fail", "module": "m"}
    ps = logger.patch_safety
    before = len(ps._records)
    layer.record_patch_outcome(sid, False)
    after = len(ps._records)
    assert after > before


def test_build_context_and_feedback_updates_weights(monkeypatch):
    results = [
        ("workflow", "w1", 0.9),
        ("enhancement", "e1", 0.8),
        ("resource", "r1", 0.7),
    ]
    retriever = DummyRetriever(results)
    ranker = DummyRankingModel()
    tracker = DummyROITracker()
    metrics = VectorMetricsDB(":memory:")

    class RiskyPatchLogger(DummyPatchLogger):
        def track_contributors(
            self,
            vector_ids,
            result,
            *,
            patch_id="",
            session_id="",
            contribution=None,
            retrieval_metadata=None,
            lines_changed=None,
            tests_passed=None,
            enhancement_name=None,
            start_time=None,
            end_time=None,
            diff=None,
            summary=None,
            outcome=None,
            error_summary=None,
        ):
            super().track_contributors(
                vector_ids,
                result,
                patch_id=patch_id,
                session_id=session_id,
                contribution=contribution,
                retrieval_metadata=retrieval_metadata,
                lines_changed=lines_changed,
                tests_passed=tests_passed,
                enhancement_name=enhancement_name,
                start_time=start_time,
                end_time=end_time,
                diff=diff,
                summary=summary,
                outcome=outcome,
                error_summary=error_summary,
            )
            if not result:
                scores = {}
                for vid, _score in vector_ids:
                    origin = vid.split(":", 1)[0]
                    scores[origin] = 0.5
                return scores
            return {}

    logger = RiskyPatchLogger(metrics, tracker)
    builder = DummyContextBuilder(retriever, ranking_model=ranker)
    layer = CognitionLayer(
        context_builder=builder,
        patch_logger=logger,
        vector_metrics=metrics,
        roi_tracker=tracker,
    )

    import sys
    import types

    sys.modules.setdefault("roi_tracker", types.SimpleNamespace(ROITracker=DummyROITracker))

    import cognition_layer as cl

    monkeypatch.setattr(cl, "_roi_tracker", tracker)
    setattr(builder, "_cognition_layer", layer)

    ctx, sid = cl.build_cognitive_context("hello", top_k=3, context_builder=builder)
    assert ctx and sid

    cl.log_feedback(sid, True, patch_id="p1", context_builder=builder)

    weights_success = metrics.get_db_weights()
    assert set(weights_success) == {"workflow", "enhancement", "resource"}
    assert all(w > 0 for w in weights_success.values())
    assert tracker.db_metrics

    calls: dict[str, object] = {}

    async def fake_schedule_backfill(*, dbs=None, **_):
        calls["dbs"] = dbs

    monkeypatch.setattr(
        "vector_service.cognition_layer.schedule_backfill", fake_schedule_backfill
    )

    ctx2, sid2 = cl.build_cognitive_context("boom", top_k=3, context_builder=builder)
    cl.log_feedback(sid2, False, patch_id="p2", context_builder=builder)

    weights_failure = metrics.get_db_weights()
    assert any(weights_failure[db] < weights_success[db] for db in weights_success)
    assert set(calls.get("dbs") or []) == {"workflow", "enhancement", "resource"}


def test_wrapper_build_and_feedback(monkeypatch):
    import sys
    import types
    from vector_service.cognition_layer import CognitionLayer
    from vector_metrics_db import VectorMetricsDB

    class DummyTracker:
        def __init__(self):
            self.metrics = None

        def update_db_metrics(self, metrics):
            self.metrics = metrics

        def update(self, *a, **k):
            pass

        def origin_db_deltas(self):
            return {}

    sys.modules.setdefault("roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))
    import cognition_layer as cl

    index = {
        "workflow": {"w1": [0.1, 0.2]},
        "enhancement": {"e1": [0.3, 0.4]},
        "resource": {"r1": [0.5, 0.6]},
    }

    class DummyContextBuilder:
        def __init__(self, index):
            self.index = index

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
            vectors = [
                ("workflow", "w1", 0.9),
                ("enhancement", "e1", 0.8),
                ("resource", "r1", 0.7),
            ]
            sid = session_id or "sid"
            stats = {"tokens": 1, "wall_time_ms": 1.0, "prompt_tokens": len(prompt.split())}
            meta = {
                "misc": [
                    {
                        "origin_db": o,
                        "vector_id": v,
                        "metadata": {"risk_score": rs},
                    }
                    for (o, v, _), rs in zip(vectors, [0.2, 0.1, 0.3])
                ]
            }
            if include_vectors:
                if return_metadata:
                    if return_stats:
                        return "context", sid, vectors, meta, stats
                    return "context", sid, vectors, meta
                if return_stats:
                    return "context", sid, vectors, stats
                return "context", sid, vectors
            if return_metadata:
                return "context", meta
            if return_stats:
                return "context", stats
            return "context"

    tracker = DummyTracker()
    metrics = VectorMetricsDB(":memory:")
    builder = DummyContextBuilder(index)
    layer = CognitionLayer(
        context_builder=builder,
        vector_metrics=metrics,
        roi_tracker=tracker,
    )
    setattr(builder, "_cognition_layer", layer)

    ctx, sid = cl.build_cognitive_context("hello world", context_builder=builder)
    assert ctx and sid

    cl.log_feedback(sid, True, patch_id="p1", context_builder=builder)
    weights_success = metrics.get_db_weights()
    assert set(weights_success) == {"workflow", "enhancement", "resource"}
    assert all(w > 0 for w in weights_success.values())

    calls: dict[str, object] = {}

    async def fake_schedule_backfill(*, dbs=None, **_):
        calls["dbs"] = dbs

    monkeypatch.setattr(
        "vector_service.cognition_layer.schedule_backfill", fake_schedule_backfill
    )

    ctx2, sid2 = cl.build_cognitive_context("again", context_builder=builder)
    assert ctx2 and sid2
    cl.log_feedback(sid2, False, patch_id="p2", context_builder=builder)

    weights_failure = metrics.get_db_weights()
    assert all(weights_failure[db] == 0.0 for db in weights_failure)
    assert set(calls.get("dbs") or []) == {"workflow", "enhancement", "resource"}
