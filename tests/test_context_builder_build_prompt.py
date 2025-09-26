import json
import pytest
from prompt_types import Prompt

from vector_service.context_builder import build_prompt


class DummyBuilder:
    prompt_score_weight = 1.0
    roi_weight = 1.0
    recency_weight = 1.0
    safety_weight = 1.0
    prompt_max_tokens = 1000

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def build_context(self, query, *, top_k=5, include_vectors=True, return_metadata=True, **kwargs):
        context = json.dumps({"code": [{"desc": "fix bug"}], "docs": [{"desc": "update docs"}]}, separators=(",", ":"))
        vectors = [("code", "v1", 0.9), ("code", "v2", 0.5), ("docs", "v3", 0.7)]
        meta = {
            "code": [
                {"desc": "fix bug", "score": 0.9, "roi": 1.0, "recency": 0.5, "risk_score": 0.1},
                {"desc": "fix bug", "score": 0.5, "roi": 0.5, "recency": 0.2, "risk_score": 0.2},
            ],
            "docs": [
                {"desc": "update docs", "score": 0.7, "roi": 0.3, "recency": 0.1, "risk_score": 0.0}
            ],
        }
        return context, "sid", vectors, meta


def test_build_prompt_enriches_metadata(monkeypatch):
    captured = {}

    def fake_pe_build(goal, *, retrieval_context=None, context_builder=None, **kwargs):
        captured["retrieval_context"] = retrieval_context
        return Prompt("built", metadata={"engine": "meta"})

    monkeypatch.setattr("prompt_engine.build_prompt", fake_pe_build)

    builder = DummyBuilder()
    prompt = build_prompt(
        "add feature",
        context_builder=builder,
        intent_metadata={"user": "intent"},
    )

    # retrieval_context deduplicates repeated snippets
    assert captured["retrieval_context"].count("fix bug") == 1
    # intent metadata merged
    assert prompt.metadata["intent"] == {"user": "intent"}
    # vector metadata fused
    assert prompt.metadata["vectors"][0] == ("code", "v1", 0.9)
    # vector confidence averaged
    assert prompt.vector_confidence == pytest.approx((0.9 + 0.5 + 0.7) / 3)


def test_build_prompt_surfaces_stack_snippets(monkeypatch):
    captured = {}

    def fake_pe_build(goal, *, retrieval_context=None, context_builder=None, **kwargs):
        captured["retrieval_context"] = retrieval_context
        return Prompt("built", metadata={})

    monkeypatch.setattr("prompt_engine.build_prompt", fake_pe_build)

    class StackBuilder(DummyBuilder):
        stack_prompt_enabled = True
        stack_prompt_limit = 1

        def build_context(self, query, *, top_k=5, include_vectors=True, return_metadata=True, **kwargs):
            context = json.dumps({"stack": []}, separators=(",", ":"))
            vectors = [("stack", "s1", 0.9), ("stack", "s2", 0.8)]
            meta = {
                "stack": [
                    {
                        "desc": "repo1/main.py [python]\nprint('hello')",
                        "score": 0.9,
                        "repo": "repo1",
                        "path": "main.py",
                        "language": "python",
                        "vector_id": "s1",
                        "origin": "stack",
                    },
                    {
                        "desc": "repo2/app.js [javascript]\nconsole.log('hi')",
                        "score": 0.8,
                        "repo": "repo2",
                        "path": "app.js",
                        "language": "javascript",
                        "vector_id": "s2",
                        "origin": "stack",
                    },
                ]
            }
            return context, "sid", vectors, meta

    builder = StackBuilder()
    prompt = build_prompt(
        "use stack",
        context_builder=builder,
        include_stack_snippets=True,
        stack_snippet_limit=1,
    )

    assert "repo1/main.py" in captured["retrieval_context"]
    assert "repo2/app.js" not in captured["retrieval_context"]
    stack_meta = prompt.metadata["stack_snippets"]
    assert len(stack_meta) == 1
    entry = stack_meta[0]
    assert entry["key"] == "stack:s1"
    assert entry["language"] == "python"
    retrieval_meta = prompt.metadata["retrieval_metadata"]
    assert "stack:s1" in retrieval_meta
    assert retrieval_meta["stack:s1"]["prompt_tokens"] > 0
