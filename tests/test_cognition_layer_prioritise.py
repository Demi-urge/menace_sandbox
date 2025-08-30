import asyncio
from vector_service.cognition_layer import CognitionLayer


class DummyVectorMetrics:
    def get_db_weights(self):
        return {}

    def log_retrieval(self, *a, **k):
        pass

    def save_session(self, *a, **k):
        pass


class DummyPatchLogger:
    def track_contributors(self, *a, **k):
        return {}


class DummyROITracker:
    def update_db_metrics(self, metrics):
        pass

    def update(self, *a, **k):
        pass


class DummyContextBuilder:
    def __init__(self):
        self.calls = []
        self.retriever = object()

    def build_context(
        self,
        prompt,
        *,
        top_k=5,
        include_vectors=False,
        session_id="",
        return_stats=False,
        return_metadata=False,
        prioritise=None,
    ):
        self.calls.append(prioritise)
        vectors = [("db", "v", 0.1)]
        stats = {"tokens": 0, "wall_time_ms": 0.0, "prompt_tokens": 0}
        sid = session_id or "sid"
        meta = {"misc": []}
        return "ctx", sid, vectors, meta, stats

    async def build_async(self, prompt, **kwargs):
        self.calls.append(kwargs.get("prioritise"))
        vectors = [("db", "v", 0.1)]
        stats = {"tokens": 0, "wall_time_ms": 0.0, "prompt_tokens": 0}
        sid = kwargs.get("session_id", "") or "sid"
        return "ctx", sid, vectors, {}, stats


def _make_layer(builder):
    return CognitionLayer(
        context_builder=builder,
        vector_metrics=DummyVectorMetrics(),
        patch_logger=DummyPatchLogger(),
        retriever=builder.retriever,  # avoid heavy initialisation
        patch_retriever=None,
        roi_tracker=DummyROITracker(),
        ranking_model=None,
    )


def test_query_prioritise_forwarded():
    builder = DummyContextBuilder()
    layer = _make_layer(builder)
    layer.query("p", prioritise="roi")
    assert builder.calls[0] == "roi"


def test_query_async_prioritise_forwarded():
    builder = DummyContextBuilder()
    layer = _make_layer(builder)
    asyncio.run(layer.query_async("p", prioritise="newest"))
    assert builder.calls[0] == "newest"
