from __future__ import annotations

import importlib
import sys
import types


def test_get_text_embeddings_is_lazy(monkeypatch):
    """The embedder is obtained only when embeddings are requested."""

    calls: list[str] = []

    class DummyEmbedder:
        def __init__(self) -> None:
            self._dim = 2

        def encode(self, texts):  # type: ignore[override]
            return [[0.1, 0.2] for _ in texts]

        def get_sentence_embedding_dimension(self) -> int:
            return self._dim

    def fake_get_embedder():
        calls.append("called")
        return DummyEmbedder()

    governed_stub = types.ModuleType("governed_embeddings")
    governed_stub.get_embedder = fake_get_embedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "governed_embeddings", governed_stub)

    st_stub = types.ModuleType("sentence_transformers")
    st_stub.SentenceTransformer = DummyEmbedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.atleast_2d = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    sys.modules.pop("vector_service.vectorizer", None)
    vectorizer_stub = types.ModuleType("vector_service.vectorizer")
    vectorizer_stub.SharedVectorService = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    sys.modules.pop("vector_service.embed_utils", None)
    embed_utils = importlib.import_module("vector_service.embed_utils")

    assert calls == []  # lazy: nothing fetched during import

    result = embed_utils.get_text_embeddings(["hello"])

    assert calls == ["called"]
    assert result == [[0.1, 0.2]]
    assert embed_utils.EMBED_DIM == 2
