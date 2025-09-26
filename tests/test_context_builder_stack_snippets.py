import logging
import sys
import types

import pytest

import menace

sys.modules.setdefault("menace_sandbox", menace)

import vector_service.context_builder as cb_mod
from config import StackDatasetConfig
from vector_service.context_builder import ContextBuilder
from vector_service.retriever import StackRetriever

if not hasattr(cb_mod, "_first_non_none"):
    cb_mod._first_non_none = lambda *values: next((value for value in values if value is not None), None)


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


class DummyStackStore:
    def __init__(self) -> None:
        self.ids = ["stack:1"]
        self.meta = [
            {
                "id": "stack:1",
                "metadata": {
                    "origin": "stack",
                    "summary": "safe snippet",
                    "language": "python",
                    "repo": "safe/repo",
                    "path": "main.py",
                },
            }
        ]
        self.vectors = [[1.0, 1.0]]

    def query(self, vector, top_k: int = 5):
        return [("stack:1", 0.1)]


class DummyVectorService:
    def __init__(self, store: DummyStackStore) -> None:
        self.vector_store = store

    def vectorise(self, kind: str, payload):
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        return [float(len(text) or 1.0), 1.0]


def _configure_stack(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, languages: set[str] | None = None) -> StackDatasetConfig:
    langs = languages or {"python"}
    stack_cfg = StackDatasetConfig(enabled=enabled, languages=set(langs))
    context_cfg = types.SimpleNamespace(
        stack=stack_cfg,
        stack_enabled=enabled,
        stack_languages=set(langs),
        stack_top_k=1,
        stack_prompt_enabled=True,
        stack_prompt_limit=2,
        stack_max_lines=3,
    )
    monkeypatch.setattr(
        cb_mod,
        "get_config",
        lambda: types.SimpleNamespace(context_builder=context_cfg, stack_dataset=stack_cfg),
    )
    return stack_cfg


def _build_stack_enabled_builder(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> ContextBuilder:
    stack_cfg = _configure_stack(monkeypatch, enabled=enabled)
    return ContextBuilder(
        retriever=DummyRetriever(),
        patch_retriever=DummyPatchRetriever(),
        stack_enabled=enabled,
        stack_languages=set(stack_cfg.languages),
        stack_top_k=1,
        stack_prompt_enabled=True,
        stack_prompt_limit=2,
        embedding_check_interval=0,
        stack_config=stack_cfg,
    )


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


def test_stack_register_respects_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _build_stack_enabled_builder(monkeypatch, enabled=False)
    store = DummyStackStore()
    vector_service = DummyVectorService(store)

    builder.register_stack_index(stack_index=store, vector_service=vector_service)

    assert builder.stack_retriever is None


def test_stack_register_creates_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _build_stack_enabled_builder(monkeypatch, enabled=True)
    store = DummyStackStore()
    vector_service = DummyVectorService(store)

    builder.register_stack_index(stack_index=store, vector_service=vector_service)

    assert builder.stack_retriever is not None
    assert builder.stack_retriever.languages == {"python"}


def test_self_coding_blends_stack_context(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _build_stack_enabled_builder(monkeypatch, enabled=True)
    monkeypatch.setattr(cb_mod, "ensure_embeddings_fresh", lambda *a, **k: None)
    builder._count_tokens = types.MethodType(lambda self, text: len(str(text).split()), builder)

    store = DummyStackStore()
    vector_service = DummyVectorService(store)
    stack_retriever = StackRetriever(
        context_builder=builder,
        stack_index=store,
        vector_service=vector_service,
        top_k=1,
        metadata_db_path=None,
        max_lines=3,
        patch_safety=builder.patch_safety,
        languages={"python"},
    )
    store.query = lambda vec, top_k=5: [("stack:1", 0.1)]
    builder.stack_retriever = stack_retriever
    builder._stack_ready_checked = True

    import menace.self_coding_engine as sce

    engine = sce.SelfCodingEngine.__new__(sce.SelfCodingEngine)
    engine.logger = logging.getLogger("test-self-coding-stack")
    engine._last_retry_trace = "trace"
    engine._last_prompt_metadata = {}
    engine._last_prompt = None

    prompt = engine.build_enriched_prompt(
        "need stack context",
        context_builder=builder,
    )

    stack_meta = prompt.metadata.get("stack_snippets", [])
    assert stack_meta and stack_meta[0]["key"].startswith("stack:")
    retrieval_meta = prompt.metadata["retrieval_metadata"]
    assert "code:c1" in retrieval_meta
    assert any(key.startswith("stack:") for key in retrieval_meta)
    assert any("safe snippet" in example for example in prompt.examples)
    assert engine._last_prompt is prompt
