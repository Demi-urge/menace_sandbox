import asyncio
import time

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def __init__(self, results=None):
        self.results = results or []

    def search(self, query, top_k=5, session_id=""):
        return list(self.results)


class DummyContextBuilder:
    def __init__(self, retriever):
        self.retriever = retriever

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
        stats = {
            "tokens": len(prompt.split()),
            "wall_time_ms": 1.0,
            "prompt_tokens": len(prompt.split()),
        }
        sid = session_id or "sid"
        ts = time.time() - 30.0
        meta = {
            "misc": [
                {
                    "origin_db": origin,
                    "vector_id": vid,
                    "metadata": {"timestamp": ts},
                }
                for origin, vid, _ in vectors
            ]
        }
        return "ctx", sid, vectors, meta, stats


class DualPatchLogger:
    def __init__(self):
        self.sync_calls = []
        self.async_calls = []
        self.roi_tracker = None

    def track_contributors(
        self,
        vector_ids,
        result,
        *,
        patch_id="",
        session_id="",
        contribution=None,
        roi_delta=None,
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
        self.sync_calls.append(
            {
                "vector_ids": vector_ids,
                "result": result,
                "patch_id": patch_id,
                "session_id": session_id,
                "contribution": contribution,
                "roi_delta": roi_delta,
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
        return {}

    async def track_contributors_async(
        self,
        vector_ids,
        result,
        *,
        patch_id="",
        session_id="",
        contribution=None,
        roi_delta=None,
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
        self.async_calls.append(
            {
                "vector_ids": vector_ids,
                "result": result,
                "patch_id": patch_id,
                "session_id": session_id,
                "contribution": contribution,
                "roi_delta": roi_delta,
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
        return {}


def _prep_layer():
    retriever = DummyRetriever([("db", "v1", 0.5)])
    builder = DummyContextBuilder(retriever)
    metrics = VectorMetricsDB(":memory:")
    layer = CognitionLayer(
        context_builder=builder,
        patch_logger=DualPatchLogger(),
        vector_metrics=metrics,
        roi_tracker=None,
    )
    return layer


def _run(async_mode):
    layer = _prep_layer()
    _, sid = layer.query("hello")
    asyncio.run(layer._record_patch_outcome_impl(sid, True, async_mode=async_mode))
    logger = layer.patch_logger
    calls = logger.async_calls if async_mode else logger.sync_calls
    assert len(calls) == 1
    call = calls[0].copy()
    call.pop("retrieval_metadata", None)
    return call


def test_record_patch_outcome_impl_sync_async_parity():
    sync_call = _run(False)
    async_call = _run(True)
    assert sync_call == async_call
