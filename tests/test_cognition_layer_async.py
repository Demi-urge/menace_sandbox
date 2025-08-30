import asyncio
import time

from vector_service.cognition_layer import CognitionLayer


class DummyContextBuilder:
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
        vectors = [(prompt, "v1", 0.5)]
        stats = {"tokens": 1, "wall_time_ms": 1.0, "prompt_tokens": len(prompt.split())}
        meta = {
            "misc": [
                {
                    "origin_db": prompt,
                    "vector_id": "v1",
                    "metadata": {"timestamp": time.time() - 30.0},
                }
            ]
        }
        sid = session_id or f"sid-{prompt}"
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

    async def build_async(
        self,
        prompt,
        *,
        top_k=5,
        include_vectors=False,
        session_id="",
        return_stats=False,
        return_metadata=False,
    ):
        await asyncio.sleep(0.05)
        return self.build_context(
            prompt,
            top_k=top_k,
            include_vectors=include_vectors,
            session_id=session_id,
            return_stats=return_stats,
            return_metadata=return_metadata,
        )


class DummyVectorMetrics:
    def log_retrieval(self, *a, **k):
        pass


class DummyPatchLogger:
    def __init__(self):
        self.sessions = []

    async def track_contributors_async(
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
        await asyncio.sleep(0.05)
        self.sessions.append(
            {
                "session_id": session_id,
                "lines_changed": lines_changed,
                "tests_passed": tests_passed,
                "enhancement_name": enhancement_name,
                "start_time": start_time,
                "end_time": end_time,
                "effort_estimate": effort_estimate,
            }
        )
        return {}


async def _run_session(layer, name):
    ctx, sid = await layer.query_async(name)
    await layer.record_patch_outcome_async(
        sid,
        True,
        lines_changed=1,
        tests_passed=True,
        enhancement_name="feat",
        timestamp=2.0,
    )
    return sid


def test_concurrent_async_usage():
    layer = CognitionLayer(
        context_builder=DummyContextBuilder(),
        vector_metrics=DummyVectorMetrics(),
        patch_logger=DummyPatchLogger(),
    )

    start = time.time()

    async def runner():
        return await asyncio.gather(
            _run_session(layer, "a"),
            _run_session(layer, "b"),
        )

    sids = asyncio.run(runner())
    duration = time.time() - start

    assert len(set(sids)) == 2
    assert [s["session_id"] for s in layer.patch_logger.sessions] == sids
    for meta in layer.patch_logger.sessions:
        assert meta["lines_changed"] == 1
        assert meta["tests_passed"] is True
        assert meta["enhancement_name"] == "feat"
        assert meta["end_time"] == 2.0
    assert duration < 0.17
