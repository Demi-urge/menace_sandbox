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
