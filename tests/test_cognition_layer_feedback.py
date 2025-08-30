import pytest
import time

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class SequentialRetriever:
    """Return a different result set on each search call."""

    def __init__(self, *result_sets):
        # each result set is a list of (origin, vector_id, score)
        self.result_sets = list(result_sets)
        self.calls = []

    def search(self, query, top_k=5, session_id=""):
        self.calls.append((query, top_k, session_id))
        return list(self.result_sets.pop(0)) if self.result_sets else []


class WeightRankingModel:
    """Ranking model that biases scores using per-origin weights."""

    def __init__(self, weights):
        self.weights = weights
        self.rank_calls = 0

    def rank(self, items):
        self.rank_calls += 1
        # items: list of (origin, vector_id, score)
        return sorted(
            items,
            key=lambda t: t[2] + self.weights.get(t[0], 0.0),
            reverse=True,
        )


class DummyContextBuilder:
    """Minimal context builder similar to the one in cognition layer tests."""

    def __init__(self, retriever, ranking_model=None):
        self.retriever = retriever
        self.ranking_model = ranking_model
        self.db_weights = {}

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
        vectors = vectors[: top_k]
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
        effort_estimate=None,
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
                "start_time": start_time,
                "end_time": end_time,
                "diff": diff,
                "summary": summary,
                "outcome": outcome,
                "effort_estimate": effort_estimate,
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


def _make_layer(result_sets, tmp_path):
    weights = {}
    retriever = SequentialRetriever(*result_sets)
    ranker = WeightRankingModel(weights)
    tracker = DummyROITracker()
    metrics = VectorMetricsDB(tmp_path / "vm.db")
    builder = DummyContextBuilder(retriever, ranking_model=ranker)
    builder.db_weights = weights
    logger = DummyPatchLogger(metrics, tracker)
    layer = CognitionLayer(
        context_builder=builder,
        patch_logger=logger,
        vector_metrics=metrics,
        roi_tracker=tracker,
    )
    return layer, retriever, ranker, tracker, metrics, logger


def test_successful_patch_boosts_future_ranking(tmp_path):
    layer, retriever, ranker, tracker, metrics, _ = _make_layer(
        [
            [("db1", "v1", 0.4)],
            [("db1", "v1", 0.2), ("db2", "v2", 0.2)],
        ],
        tmp_path,
    )

    _, sid = layer.query("first")
    layer.record_patch_outcome(sid, True, contribution=1.0)

    assert metrics.get_db_weights()["db1"] > 0
    assert tracker.db_metrics[0]["db1"]["roi"] == pytest.approx(1.0)

    _, sid2 = layer.query("second", top_k=2)
    vectors = layer._session_vectors[sid2]
    assert vectors[0][0] == "db1"


def test_risky_vector_downranked_after_feedback(tmp_path):
    layer, retriever, ranker, tracker, metrics, logger = _make_layer(
        [
            [("risky", "vr", 0.3)],
            [("risky", "vr", 0.3), ("safe", "vs", 0.3)],
        ],
        tmp_path,
    )

    _, sid = layer.query("risk")
    layer._retrieval_meta[sid]["risky:vr"]["alignment_severity"] = 5.0
    layer.record_patch_outcome(sid, False, contribution=1.0)

    assert metrics.get_db_weights()["risky"] == pytest.approx(0.0)
    assert logger.calls[0]["retrieval_metadata"]["risky:vr"]["alignment_severity"] == 5.0
    assert tracker.db_metrics[0]["risky"]["regret_rate"] == pytest.approx(1.0)
