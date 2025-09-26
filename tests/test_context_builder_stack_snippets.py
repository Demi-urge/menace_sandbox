import types

from vector_service.context_builder import ContextBuilder


class DummyRetriever:
    def __init__(self) -> None:
        self.max_alert_severity = 1.0
        self.max_alerts = 5
        self.license_denylist = set()
        self.vector_service = None

    def search(self, query, *, top_k=5, session_id=None, max_alert_severity=None):
        return [
            {
                "origin_db": "code",
                "record_id": "c1",
                "score": 0.8,
                "text": "internal fix",
                "metadata": {"redacted": True, "summary": "internal fix"},
            }
        ]


class DummyPatchRetriever:
    def __init__(self) -> None:
        self.enhancement_weight = 0.0
        self.roi_tag_weights = {}
        self.vector_service = None

    def search(self, query, *, top_k=5):
        return []


class FakeStackRetriever:
    def __init__(self, patch_safety) -> None:
        self.namespace = "stack"
        self.patch_safety = patch_safety
        self.max_alert_severity = patch_safety.max_alert_severity
        self.max_alerts = patch_safety.max_alerts
        self.license_denylist = set()

    def retrieve(self, query, *, k=None, exclude_tags=None):
        safe_meta = {
            "redacted": True,
            "repo": "safe/repo",
            "path": "main.py",
            "language": "python",
            "summary": "safe snippet",
        }
        risky_meta = {
            "redacted": True,
            "repo": "risk/repo",
            "path": "danger.py",
            "language": "python",
            "summary": "risky snippet",
            "semantic_alerts": ["alert"] * 6,
        }
        return [
            {
                "origin_db": "stack",
                "record_id": "safe",
                "score": 0.9,
                "text": "safe snippet",
                "metadata": safe_meta,
            },
            {
                "origin_db": "stack",
                "record_id": "risky",
                "score": 0.8,
                "text": "risky snippet",
                "metadata": risky_meta,
            },
        ]

    def is_index_stale(self) -> bool:
        return False


def test_stack_snippets_merge_and_filter(monkeypatch):
    monkeypatch.setattr(
        "vector_service.context_builder.ensure_embeddings_fresh",
        lambda *a, **k: None,
    )
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        patch_retriever=DummyPatchRetriever(),
    )
    builder._count_tokens = types.MethodType(lambda self, text: len(str(text).split()), builder)
    builder.prompt_max_tokens = 50
    builder.stack_enabled = True
    builder.stack_prompt_enabled = True
    builder.stack_prompt_limit = 2
    builder.stack_top_k = 2
    builder.stack_languages = {"python"}
    builder._stack_ready_checked = True
    builder.stack_retriever = FakeStackRetriever(builder.patch_safety)

    prompt = builder.build_prompt(
        "need stack context",
        include_stack_snippets=True,
        top_k=2,
        stack_snippet_limit=2,
    )

    stack_meta = prompt.metadata["stack_snippets"]
    assert len(stack_meta) == 1
    assert stack_meta[0]["key"] == "stack:safe"
    retrieval_meta = prompt.metadata["retrieval_metadata"]
    assert "stack:safe" in retrieval_meta
    assert "stack:risky" not in retrieval_meta
    assert retrieval_meta["stack:safe"]["prompt_tokens"] > 0
    assert any("internal fix" in example for example in prompt.examples)
