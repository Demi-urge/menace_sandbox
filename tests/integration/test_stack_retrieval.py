from __future__ import annotations

from typing import Any, Iterable, List, Mapping

import json

import pytest

from prompt_types import Prompt
from vector_service.context_builder import ContextBuilder


class DummyRetriever:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Any]] = []

    def search(self, query: str, **_: Any) -> List[Mapping[str, Any]]:
        self.calls.append(("search", query))
        return []

    def embed_query(self, query: str) -> List[float]:
        self.calls.append(("embed", query))
        return [0.42, 0.17, 0.33]


class DummyPatchRetriever:
    def search(self, query: str, **_: Any) -> List[Mapping[str, Any]]:
        return []


class FakeStackRetriever:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Any]] = []
        self.top_k = 1
        self.max_alert_severity = 1.0
        self.max_alerts = 5
        self.license_denylist: set[str] = set()
        self.roi_tag_weights: Mapping[str, float] = {}

    def embed_query(self, query: str) -> List[float]:
        self.calls.append(("embed_query", query))
        return [0.1, 0.2, 0.3]

    def retrieve(
        self,
        embedding: Iterable[float],
        *,
        k: int | None = None,
        languages: Iterable[str] | None = None,
        max_lines: int | None = None,
        **_: Any,
    ) -> List[Mapping[str, Any]]:
        self.calls.append(
            (
                "retrieve",
                tuple(float(x) for x in embedding),
                k,
                tuple(languages or ()),
                max_lines,
            )
        )
        snippet = "Stack helper snippet text from external source"
        metadata = {
            "summary": snippet,
            "desc": snippet,
            "redacted": True,
            "repo": "octo/demo",
            "path": "src/app.py",
            "language": "Python",
            "license": "MIT",
        }
        return [
            {
                "identifier": "stack-123",
                "score": 0.91,
                "text": snippet,
                "metadata": metadata,
                "repo": metadata["repo"],
                "path": metadata["path"],
                "language": metadata["language"],
                "license": metadata["license"],
                "origin_db": "stack",
            }
        ]


@pytest.fixture(autouse=True)
def _no_embedding_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vector_service.context_builder.ensure_embeddings_fresh", lambda *_: None
    )


def test_stack_hits_contribute_to_prompt() -> None:
    stack = FakeStackRetriever()
    builder = ContextBuilder(
        retriever=DummyRetriever(),
        patch_retriever=DummyPatchRetriever(),
        stack_retriever=stack,
        stack_config={
            "enabled": True,
            "top_k": 1,
            "summary_tokens": 120,
            "text_max_tokens": 320,
            "ensure_before_search": False,
        },
        db_weights={"code": 1.0, "stack": 1.0},
    )

    bundles = builder._retrieve_stack_hits("Need stack example")
    assert bundles and isinstance(bundles[0], Mapping)
    bundle = bundles[0]
    meta = bundle["metadata"]
    assert "Stack helper snippet text" in bundle.get("text", "")
    assert meta["repo"] == "octo/demo"
    assert meta["path"] == "src/app.py"
    assert meta["language"] == "Python"
    assert meta["license"] == "MIT"
    assert meta["redacted"] is True
    assert bundle["score"] > 0
    assert stack.calls and stack.calls[0][0] == "embed_query"

    prompt = builder.build_prompt("Need stack example", top_k=1)
    assert isinstance(prompt, Prompt)
