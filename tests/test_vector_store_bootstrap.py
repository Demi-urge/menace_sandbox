import logging
import sys
from types import SimpleNamespace

import numpy as np


def _install_lightweight_stubs() -> None:
    chroma_stub = SimpleNamespace(PersistentClient=lambda *_, **__: SimpleNamespace(
        get_or_create_collection=lambda *_a, **_k: SimpleNamespace(
            add=lambda *a, **k: None, count=lambda: 0, query=lambda *a, **k: {}
        )
    ))
    qdrant_models = SimpleNamespace(Distance=SimpleNamespace(COSINE="cosine"), VectorParams=object, PointStruct=object)
    qdrant_stub = SimpleNamespace(
        QdrantClient=lambda *a, **k: SimpleNamespace(
            recreate_collection=lambda *a, **k: None,
            upsert=lambda *a, **k: None,
            get_collection=lambda *a, **k: None,
            search=lambda *a, **k: [],
        ),
        http=SimpleNamespace(models=qdrant_models),
    )

    sys.modules.setdefault("chromadb", chroma_stub)  # type: ignore
    sys.modules.setdefault("qdrant_client", qdrant_stub)  # type: ignore
    sys.modules.setdefault("qdrant_client.http", SimpleNamespace(models=qdrant_models))  # type: ignore
    sys.modules.setdefault("qdrant_client.http.models", qdrant_models)  # type: ignore
    sys.modules.setdefault(
        "faiss",
        SimpleNamespace(IndexFlatL2=lambda dim: SimpleNamespace(dim=dim), read_index=lambda *_a, **_k: None, write_index=lambda *_a, **_k: None),
    )


_install_lightweight_stubs()

from vector_service import vector_store


def test_faiss_lazy_defaults_under_bootstrap(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("MENACE_BOOTSTRAP_FAST", "1")

    load_calls: list[str] = []

    class DummyIndex:
        def __init__(self, dim: int):
            self.dim = dim

        def add(self, _vec: np.ndarray) -> None:  # pragma: no cover - not used here
            return None

        def search(self, _vec: np.ndarray, top_k: int):
            distances = np.zeros((1, top_k), dtype="float32")
            indices = np.zeros((1, top_k), dtype="int64")
            return distances, indices

    def _read_index(path: str) -> DummyIndex:
        load_calls.append(path)
        return DummyIndex(dim=3)

    dummy_faiss = SimpleNamespace(
        IndexFlatL2=lambda dim: DummyIndex(dim),
        read_index=_read_index,
        write_index=lambda *_a, **_k: None,
    )

    monkeypatch.setattr(vector_store, "faiss", dummy_faiss)
    monkeypatch.setattr(vector_store, "np", np)

    index_path = tmp_path / "faiss.index"
    index_path.touch()

    store = vector_store.create_vector_store(
        dim=3,
        path=index_path,
        backend="faiss",
        lazy=None,
    )

    assert isinstance(store, vector_store.FaissVectorStore)
    assert store.lazy is True
    assert load_calls == []
    assert any(record.message == "vector_store.lazy.defaulted" for record in caplog.records)

    store.query([0.0, 0.0, 0.0], top_k=1)

    assert load_calls == [str(index_path)]
