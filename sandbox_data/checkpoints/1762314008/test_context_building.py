import json
from vector_service.context_builder import ContextBuilder


class DummyRetriever:
    def search(self, query, top_k=5, session_id="", **kwargs):
        return [
            {
                "origin_db": "bot",
                "record_id": "a",
                "score": 0.5,
                "metadata": {"name": "a", "roi": 0.5},
            },
            {
                "origin_db": "bot",
                "record_id": "b",
                "score": 0.6,
                "metadata": {"name": "b", "roi": 0.0},
            },
        ]


def test_context_builder_orders_by_metric():
    builder = ContextBuilder(retriever=DummyRetriever())
    ctx = builder.build_context("hello", top_k=2)
    data = json.loads(ctx)
    ids = [b["id"] for b in data["bots"]]
    assert ids == ["a", "b"]
