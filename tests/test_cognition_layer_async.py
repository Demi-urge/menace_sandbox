import asyncio
import time

import asyncio
import time

from vector_service.cognition_layer import CognitionLayer


class DummyContextBuilder:
    def build_context(self, prompt, *, top_k=5, include_vectors=False, session_id="", return_stats=False):
        vectors = [(prompt, "v1", 0.5)]
        stats = {"tokens": 1, "wall_time_ms": 1.0, "prompt_tokens": len(prompt.split())}
        if include_vectors:
            if return_stats:
                return "ctx", session_id or f"sid-{prompt}", vectors, stats
            return "ctx", session_id or f"sid-{prompt}", vectors
        if return_stats:
            return "ctx", stats
        return "ctx"

    async def build_async(self, prompt, *, top_k=5, include_vectors=False, session_id="", return_stats=False):
        await asyncio.sleep(0.05)
        return self.build_context(
            prompt,
            top_k=top_k,
            include_vectors=include_vectors,
            session_id=session_id,
            return_stats=return_stats,
        )


class DummyVectorMetrics:
    def log_retrieval(self, *a, **k):
        pass


class DummyPatchLogger:
    def __init__(self):
        self.sessions = []

    async def track_contributors_async(self, vector_ids, result, *, patch_id="", session_id="", contribution=None, retrieval_metadata=None):
        await asyncio.sleep(0.05)
        self.sessions.append(session_id)


async def _run_session(layer, name):
    ctx, sid = await layer.query_async(name)
    await layer.record_patch_outcome_async(sid, True)
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
    assert layer.patch_logger.sessions == sids
    assert duration < 0.17
