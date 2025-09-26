from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import vector_service.context_builder as cb_mod
import vector_service.stack_ingestor as stack_ingestor_mod

if not hasattr(cb_mod, "_first_non_none"):
    cb_mod._first_non_none = lambda *values: next((value for value in values if value is not None), None)

ContextBuilder = cb_mod.ContextBuilder


class DummyVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)


class DummyStackRetriever:
    def __init__(self, store: DummyVectorStore) -> None:
        self.stack_index = store
        self.vector_store = store
        self.vector_service = SimpleNamespace(vector_store=store)
        self.metadata_db_path = None


def test_context_builder_ingest_updates_stack_metadata(monkeypatch, tmp_path):
    base_store = DummyVectorStore(tmp_path / "existing.index")
    retriever = DummyStackRetriever(base_store)
    monkeypatch.setattr(cb_mod, "PatchRetriever", lambda *args, **kwargs: SimpleNamespace())
    builder = ContextBuilder(stack_retriever=retriever, stack_enabled=True)
    builder.stack_cache_dir = Path(tmp_path)

    new_store = DummyVectorStore(tmp_path / "refreshed.index")
    metadata_path = tmp_path / "stack-meta.db"
    captured: dict[str, object] = {}

    class StubIngestor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.metadata_store = SimpleNamespace(path=metadata_path)
            self.vector_service = kwargs.get("vector_service")
            self.vector_store = new_store

        def ingest(self, resume: bool = True, limit: int | None = None) -> int:
            return 7

    monkeypatch.setattr(stack_ingestor_mod, "StackIngestor", StubIngestor)
    monkeypatch.setattr(cb_mod, "StackIngestor", StubIngestor, raising=False)

    processed = builder._ingest_stack_embeddings(resume=False, limit=5)

    assert processed == 7
    assert captured["vector_service"] is retriever.vector_service
    assert captured["vector_store"] is base_store
    assert builder.stack_metadata_path == metadata_path
    assert retriever.metadata_db_path == metadata_path
    assert builder.stack_index_path == new_store.path
    assert retriever.stack_index is new_store
    assert retriever.vector_store is new_store
    assert builder._stack_last_ingest == 7
