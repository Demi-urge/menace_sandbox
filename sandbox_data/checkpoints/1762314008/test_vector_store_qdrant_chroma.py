import pytest
import vector_service.vector_store as vs


@pytest.mark.parametrize(
    "backend,cls,mod_name",
    [
        ("qdrant", vs.QdrantVectorStore, "qdrant_client"),
        ("chroma", vs.ChromaVectorStore, "chromadb"),
    ],
)
def test_vector_store_roundtrip(tmp_path, backend, cls, mod_name):
    pytest.importorskip(mod_name)
    store_path = tmp_path / backend
    store = vs.create_vector_store(3, store_path, backend=backend)
    assert isinstance(store, cls)
    store.add("test", "1", [0.0, 0.0, 1.0])
    res = store.query([0.0, 0.0, 1.0], top_k=1)
    assert res and res[0][0] == "1"
    # ensure data persists via load
    if hasattr(store, "client"):
        try:
            store.client.close()
        except Exception:
            pass
    store2 = cls(dim=3, path=store_path)
    res2 = store2.query([0.0, 0.0, 1.0], top_k=1)
    assert res2 and res2[0][0] == "1"
