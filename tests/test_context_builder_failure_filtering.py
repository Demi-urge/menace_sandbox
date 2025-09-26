import json
from vector_service.context_builder import (
    ContextBuilder,
    record_failed_tags,
    _FAILED_TAG_CACHE,
)


class _StubPatchRetriever:
    def __init__(self, *args, **kwargs) -> None:
        self.enhancement_weight = 0.0
        self.roi_tag_weights = {}
        self.vector_service = None

    def search(self, *_a, **_k):
        return []


class DummyRetriever:
    def search(self, *_a, **_k):
        return [
            {
                "origin_db": "information",
                "record_id": 1,
                "score": 1.0,
                "text": "bad",
                "metadata": {"redacted": True, "tags": ["skip"], "strategy_hash": "deadbeef"},
            },
            {
                "origin_db": "information",
                "record_id": 2,
                "score": 0.5,
                "text": "good",
                "metadata": {"redacted": True},
            },
        ]


def _make_builder(monkeypatch):
    monkeypatch.setattr(
        "vector_service.context_builder.ensure_embeddings_fresh",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "vector_service.context_builder.PatchRetriever",
        _StubPatchRetriever,
    )
    builder = ContextBuilder(retriever=DummyRetriever())
    builder._stack_ready_checked = True
    return builder


def test_context_builder_filters_excluded_tags(monkeypatch):
    builder = _make_builder(monkeypatch)
    ctx = builder.query("q", exclude_tags=["skip"])
    data = json.loads(ctx)
    assert "information" in data
    infos = data["information"]
    assert len(infos) == 1
    assert infos[0]["id"] == 2


def test_exclude_failed_strategies(monkeypatch):
    builder = _make_builder(monkeypatch)
    builder.exclude_failed_strategies(["deadbeef"])
    ctx = builder.query("q")
    data = json.loads(ctx)
    infos = data["information"]
    assert len(infos) == 1
    assert infos[0]["id"] == 2


def test_record_failed_tags_filters_future_queries(monkeypatch):
    try:
        _FAILED_TAG_CACHE.clear()
    except Exception:
        pass
    record_failed_tags(["skip"])
    builder = _make_builder(monkeypatch)
    ctx = builder.query("q")
    data = json.loads(ctx)
    infos = data["information"]
    assert len(infos) == 1
    assert infos[0]["id"] == 2
