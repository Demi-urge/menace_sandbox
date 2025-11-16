from __future__ import annotations

from pathlib import Path

import yaml

from config import ContextBuilderConfig
from vector_service import context_builder as cb_module
from vector_service.context_builder import ContextBuilder


def test_stack_context_yaml_parses_defaults():
    config_path = Path(__file__).resolve().parents[1] / "config" / "stack_context.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = ContextBuilderConfig.model_validate(data["context_builder"])

    assert cfg.stack is not None
    assert cfg.stack.cache.index_path.endswith("stack_vectors")
    assert cfg.stack.ingestion.batch_size is None
    assert cfg.stack.retrieval.top_k == 50
    assert cfg.stack_dataset.ingestion.languages == []


def test_context_builder_ingest_stack_documents_uses_overrides(monkeypatch):
    class DummyRetriever:
        def search(self, *_args, **_kwargs):
            return []

    calls: dict[str, object] = {}

    class DummyStreamer:
        def __init__(self, *_, **__):
            pass

        @classmethod
        def from_environment(cls, **overrides):
            calls["overrides"] = overrides
            return cls()

        def process(self, limit=None, continuous=False):
            calls["limit"] = limit
            calls["continuous"] = continuous
            return 3

        def stop(self):
            calls["stopped"] = True

    class DummyPatchRetriever:
        def __init__(self, *_, **__):
            self.roi_tag_weights = {}

        def search(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(cb_module, "PatchRetriever", DummyPatchRetriever)
    monkeypatch.setattr(cb_module, "_StackDatasetStreamer", DummyStreamer)
    monkeypatch.setattr(cb_module, "_ensure_stack_background", None)
    monkeypatch.setattr(cb_module, "ensure_embeddings_fresh", lambda dbs: calls.setdefault("fresh", list(dbs)))

    builder = ContextBuilder(
        retriever=DummyRetriever(),
        stack_config={
            "enabled": True,
            "dataset_name": "bigcode/the-stack-v2-dedup",
            "split": "train",
            "top_k": 7,
            "ingestion_enabled": True,
            "ingestion_batch_limit": 5,
            "ingestion": {
                "languages": ["python"],
                "max_document_lines": 250,
                "chunk_overlap": 32,
                "streaming": True,
                "batch_size": 40,
            },
            "cache": {
                "index_path": "/tmp/index.faiss",
                "metadata_path": "/tmp/stack.db",
            },
        },
    )

    embedded = builder.ingest_stack_documents()

    assert embedded == 3
    assert calls["limit"] == 5
    assert calls["fresh"] == ["stack"]
    assert calls["overrides"]["allowed_languages"] == ["python"]
    assert calls["overrides"]["max_lines"] == 250
    assert calls["overrides"]["chunk_overlap"] == 32
    assert calls["overrides"]["streaming_enabled"] is True
    assert calls["overrides"]["batch_size"] == 40
    assert calls["overrides"]["vector_store_path"] == "/tmp/index.faiss"
    assert calls["overrides"]["metadata_path"] == "/tmp/stack.db"
    assert builder.stack_retrieval_limit == 7
    assert builder.stack_languages == ("python",)
    assert builder.stack_ingestion_batch_size == 40
    assert builder.stack_index_path == "/tmp/index.faiss"
    assert builder.stack_metadata_path == "/tmp/stack.db"
