import types
from pathlib import Path

import vector_service.vector_store as vs


def test_create_vector_store_fallback_to_annoy(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "faiss", None)
    store = vs.create_vector_store(3, tmp_path / "idx.ann", backend="faiss")
    assert isinstance(store, vs.AnnoyVectorStore)
