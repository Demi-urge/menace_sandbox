from vector_service.context_builder import ContextBuilder

class DummyPatchSafety:
    def __init__(self):
        self.max_alert_severity = 1.0
        self.max_alerts = 5
        self.license_denylist = set()
    def evaluate(self, *_, **__):
        return True, 0.0, {}


def test_context_respects_max_tokens():
    class DummyRetriever:
        def search(self, query, top_k=5, **_):
            text = "word " * 100
            return [
                {
                    "origin_db": "error",
                    "record_id": 1,
                    "score": 1.0,
                    "metadata": {"id": 1, "message": text, "frequency": 1},
                }
            ]

    builder = ContextBuilder(
        retriever=DummyRetriever(),
        patch_safety=DummyPatchSafety(),
        max_tokens=20,
    )
    _, stats = builder.build_context("q", return_stats=True)
    assert stats["tokens"] <= 20


def test_summary_cached():
    calls = {"count": 0}

    def summarise(text: str) -> str:
        calls["count"] += 1
        return text[:10]

    class DummyRetriever:
        def search(self, query, top_k=5, **_):
            meta = {
                "id": 1,
                "summary": "a" * 50,
                "diff": "line1\nline2\nline3",
                "patch_id": 42,
            }
            return [
                {
                    "origin_db": "patch",
                    "record_id": 1,
                    "score": 1.0,
                    "metadata": meta,
                }
            ]

    builder = ContextBuilder(
        retriever=DummyRetriever(),
        summariser=summarise,
        patch_safety=DummyPatchSafety(),
    )
    builder.build_context("q")
    first_calls = calls["count"]
    builder.build_context("q")
    assert calls["count"] == first_calls
