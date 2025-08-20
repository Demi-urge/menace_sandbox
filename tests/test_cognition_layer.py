import types
import pytest

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


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
        if include_vectors:
            if return_stats:
                return "ctx", sid, vectors, stats
            return "ctx", sid, vectors
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
    ):
        self.calls.append(
            {
                "vector_ids": vector_ids,
                "result": result,
                "patch_id": patch_id,
                "session_id": session_id,
                "contribution": contribution,
                "retrieval_metadata": retrieval_metadata,
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


def test_query_and_record_patch_outcome_updates_metrics_and_ranking():
    results = [("db2", "v2", 0.2), ("db1", "v1", 0.9)]
    layer, retriever, ranker, tracker, metrics, logger = _make_layer(results)

    ctx, sid = layer.query("hello world")
    assert ctx == "ctx"
    assert ranker.rank_calls == 1

    rows = metrics.conn.execute(
        "SELECT session_id, vector_id, contribution, win, regret FROM vector_metrics"
        " WHERE event_type='retrieval'"
    ).fetchall()
    assert len(rows) == 2
    assert all(row[0] == sid for row in rows)
    assert all(row[2] == 0.0 and row[3] is None and row[4] is None for row in rows)

    layer.record_patch_outcome(sid, True, contribution=1.0)

    assert len(logger.calls) == 1
    assert logger.calls[0]["session_id"] == sid
    assert logger.calls[0]["vector_ids"][0][0] == "db1:v1"

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
