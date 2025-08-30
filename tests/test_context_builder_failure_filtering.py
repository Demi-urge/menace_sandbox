import json
from vector_service.context_builder import ContextBuilder


class DummyRetriever:
    def search(self, *_a, **_k):
        return [
            {
                "origin_db": "information",
                "record_id": 1,
                "score": 1.0,
                "text": "bad",
                "metadata": {"redacted": True, "tags": ["skip"]},
            },
            {
                "origin_db": "information",
                "record_id": 2,
                "score": 0.5,
                "text": "good",
                "metadata": {"redacted": True},
            },
        ]


def test_context_builder_filters_excluded_tags():
    builder = ContextBuilder(retriever=DummyRetriever())
    ctx = builder.query("q", exclude_tags=["skip"])
    data = json.loads(ctx)
    assert "information" in data
    infos = data["information"]
    assert len(infos) == 1
    assert infos[0]["id"] == 2
